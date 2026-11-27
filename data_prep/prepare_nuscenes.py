"""Preparación de datos de nuScenes (se corre UNA sola vez, antes de medir).

Lee nuScenes con su biblioteca (nuscenes-devkit) y genera los archivos que usan
los nodos: objetos.csv, recortes.npy, recortes_meta.csv, clases.json y VERSION.
Así ningún trabajo de entrenamiento tiene que leer nuScenes.

Uso:
  python data_prep/prepare_nuscenes.py --dataroot /ruta/a/nuscenes --out data/prepared

La ruta debe contener v1.0-mini/ (las tablas JSON), samples/ y sweeps/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from output import category_2levels, category_group, write_prepared  # noqa: E402

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def bbox_2d(box, K, img_w, img_h):
    """Caja 2D (x0, y0, x1, y1) de la caja 3D proyectada a la cámara, recortada a la imagen."""
    from nuscenes.utils.geometry_utils import view_points
    corners = box.corners()                      # (3, 8) en coordenadas de la cámara
    corners = corners[:, corners[2, :] > 0.1]    # solo esquinas delante de la cámara
    if corners.shape[1] == 0:
        return None
    pts = view_points(corners, K, normalize=True)[:2]
    x0, y0 = np.clip(pts.min(axis=1), 0, [img_w, img_h])
    x1, y1 = np.clip(pts.max(axis=1), 0, [img_w, img_h])
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def square_crop(img: Image.Image, bb, margin: float, size: int) -> np.ndarray:
    """Recorte cuadrado centrado en la caja (con margen), redimensionado a size x size."""
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    box = (int(round(cx - side / 2)), int(round(cy - side / 2)),
           int(round(cx + side / 2)), int(round(cy + side / 2)))
    crop = img.crop(box)   # PIL rellena con negro lo que cae fuera de la imagen
    return np.asarray(crop.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_datos.json"))
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility

    with open(args.config) as f:
        cfg = json.load(f)
    split_of = {s: part for part, scenes in cfg["particion"].items() for s in scenes}
    size, min_side, margin = cfg["tam_recorte"], cfg["min_lado_recorte_px"], cfg["margen_recorte"]

    # La partición debe respetar la separación oficial de nuScenes:
    # test = mini_val oficial; train + val = mini_train oficial.
    from nuscenes.utils.splits import create_splits_scenes
    oficial = create_splits_scenes()
    pref = "mini_" if "mini" in cfg["version_nuscenes"] else ""
    p = cfg["particion"]
    if sorted(p["test"]) != sorted(oficial[f"{pref}val"]) or \
            sorted(p["train"] + p["val"]) != sorted(oficial[f"{pref}train"]):
        sys.exit("La partición de config_datos.json no coincide con la separación oficial de nuScenes")

    nusc = NuScenes(version=cfg["version_nuscenes"], dataroot=args.dataroot, verbose=True)
    missing = set(split_of) - {s["name"] for s in nusc.scene}
    if missing:
        sys.exit(f"Escenas de la partición que no están en {cfg['version_nuscenes']}: {sorted(missing)}")

    rows, best_crop = [], {}   # best_crop[ann_token] = (area, crop, scene, split, group)
    for scene in nusc.scene:
        if scene["name"] not in split_of:
            print(f"(se omite {scene['name']}: no está en la partición)")
            continue
        split = split_of[scene["name"]]
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])

            # 1) Tabla de objetos
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                w, l, h = ann["size"]
                dist = float(np.linalg.norm(np.array(ann["translation"][:2]) - np.array(ego["translation"][:2])))
                vel = nusc.box_velocity(ann_token)
                vx, vy = (float(vel[0]), float(vel[1])) if not np.any(np.isnan(vel[:2])) else (np.nan, np.nan)
                rows.append({
                    "annotation_token": ann_token, "sample_token": token,
                    "scene": scene["name"], "split": split,
                    "category_full": ann["category_name"],
                    "category": category_2levels(ann["category_name"]),
                    "category_group": category_group(ann["category_name"], cfg),
                    "width": w, "length": l, "height": h, "volume": w * l * h,
                    "distance": dist,
                    "visibility": int(nusc.get("visibility", ann["visibility_token"])["token"]),
                    "num_lidar_pts": ann["num_lidar_pts"], "num_radar_pts": ann["num_radar_pts"],
                    "vel_x": vx, "vel_y": vy,
                    "speed": float(np.hypot(vx, vy)) if not np.isnan(vx) else np.nan,
                })

            # 2) Recortes: para cada objeto, el de la cámara donde se ve más grande
            for cam in CAMS:
                path, boxes, K = nusc.get_sample_data(sample["data"][cam], box_vis_level=BoxVisibility.ANY)
                if not boxes:
                    continue
                img = Image.open(path).convert("RGB")
                for box in boxes:
                    bb = bbox_2d(box, K, img.width, img.height)
                    if bb is None:
                        continue
                    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                    if min(bw, bh) < min_side:
                        continue
                    area = bw * bh
                    group = category_group(box.name, cfg)
                    if group not in cfg["clases_cnn"]:
                        continue
                    if area > best_crop.get(box.token, (0,))[0]:
                        best_crop[box.token] = (area, square_crop(img, bb, margin, size),
                                                scene["name"], split, group)
            token = sample["next"]
        print(f"{scene['name']} ({split}): listo")

    objetos = pd.DataFrame(rows)
    tokens = sorted(best_crop)
    recortes = np.stack([best_crop[t][1] for t in tokens])
    meta = pd.DataFrame({"annotation_token": tokens,
                         "scene": [best_crop[t][2] for t in tokens],
                         "split": [best_crop[t][3] for t in tokens],
                         "category_group": [best_crop[t][4] for t in tokens]})
    prefix = cfg["version_nuscenes"].replace("v1.0-", "")
    write_prepared(args.out, objetos, recortes, meta, cfg, prefix=prefix)


if __name__ == "__main__":
    main()
