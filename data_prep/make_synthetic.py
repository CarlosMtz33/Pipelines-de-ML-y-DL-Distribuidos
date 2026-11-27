"""Genera datos SINTÉTICOS con el mismo formato que prepare_nuscenes.py.

Sirve para probar el sistema completo (y la demo de fallas) sin tener nuScenes
descargado. NO se usa para el reporte: las mediciones se hacen con nuScenes.

Uso:
  python data_prep/make_synthetic.py --out data/synthetic --objects 15000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from output import category_2levels, category_group, write_prepared  # noqa: E402

CATS = [("vehicle.car", 0.38, (1.9, 4.6, 1.7)), ("vehicle.truck", 0.05, (2.5, 7.0, 3.0)),
        ("vehicle.bus.rigid", 0.04, (2.9, 11.0, 3.4)), ("vehicle.motorcycle", 0.04, (0.8, 2.1, 1.4)),
        ("vehicle.bicycle", 0.03, (0.6, 1.7, 1.3)), ("human.pedestrian.adult", 0.26, (0.7, 0.7, 1.75)),
        ("movable_object.barrier", 0.12, (2.5, 0.5, 1.0)), ("movable_object.trafficcone", 0.08, (0.4, 0.4, 1.0))]


def draw(group: str, rng, size: int) -> np.ndarray:
    """Una figura distinta por clase, con ruido, para que la CNN tenga algo que aprender."""
    img = rng.integers(40, 120, size=(size, size, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size] / size
    color = rng.uniform(60, 255, size=3)

    def rect(cy, h, w):
        return (np.abs(yy - cy) < h / 2) & (np.abs(xx - 0.5) < w / 2)

    def circle(cx, cy, r):
        return (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2

    if group == "carro":         # rectángulo ancho y bajo
        mask = rect(0.6, rng.uniform(0.25, 0.4), rng.uniform(0.6, 0.85))
    elif group == "autobus":     # rectángulo grande, casi toda la imagen
        mask = rect(0.5, rng.uniform(0.6, 0.8), rng.uniform(0.8, 0.95))
    elif group == "peaton":      # elipse alta
        mask = ((xx - 0.5) / rng.uniform(0.1, 0.18)) ** 2 + ((yy - 0.5) / rng.uniform(0.35, 0.45)) ** 2 < 1
    elif group == "bicicleta":   # dos ruedas separadas
        r = rng.uniform(0.15, 0.2)
        mask = circle(0.28, 0.65, r) | circle(0.72, 0.65, r)
    elif group == "moto":        # dos ruedas juntas + cuerpo
        r = rng.uniform(0.12, 0.16)
        mask = circle(0.32, 0.7, r) | circle(0.68, 0.7, r) | rect(0.5, 0.2, 0.5)
    else:                        # triángulo / cono
        mask = (yy > 0.2) & (np.abs(xx - 0.5) < (yy - 0.2) * rng.uniform(0.3, 0.6))
    img[mask] = color
    img += rng.normal(0, 25, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--objects", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_datos.json"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    rng = np.random.default_rng(args.seed)
    scenes = [(s, part) for part, ss in cfg["particion"].items() for s in ss]
    names, probs, sizes = zip(*CATS)
    rows, crops, meta = [], [], []
    for i in range(args.objects):
        scene, split = scenes[rng.integers(len(scenes))]
        c = rng.choice(len(names), p=np.array(probs) / sum(probs))
        w, l, h = np.array(sizes[c]) * rng.uniform(0.85, 1.15, size=3)
        dist = rng.uniform(2, 60)
        vis = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.15, 0.2, 0.5]))
        lidar = max(0, int(rng.poisson(max(0.1, 4000 * w * h / dist ** 2 * (0.25 + 0.25 * vis)))))
        radar = int(rng.poisson(0.5 + 3 * (c < 3)))
        speed = abs(rng.normal(5 if c < 3 else 1, 2)) if c < 4 else 0.0
        ang = rng.uniform(0, 2 * np.pi)
        tok = f"syn{i:06d}"
        group = category_group(names[c], cfg)
        rows.append({"annotation_token": tok, "sample_token": f"s{i // 30:05d}", "scene": scene,
                     "split": split, "category_full": names[c], "category": category_2levels(names[c]),
                     "category_group": group, "width": w, "length": l, "height": h,
                     "volume": w * l * h, "distance": dist, "visibility": vis,
                     "num_lidar_pts": lidar, "num_radar_pts": radar,
                     "vel_x": speed * np.cos(ang), "vel_y": speed * np.sin(ang), "speed": speed})
        if rng.random() < 0.8:   # no todos los objetos se ven en alguna cámara
            crops.append(draw(group, rng, cfg["tam_recorte"]))
            meta.append({"annotation_token": tok, "scene": scene, "split": split, "category_group": group})
    write_prepared(args.out, pd.DataFrame(rows), np.stack(crops), pd.DataFrame(meta), cfg, prefix="synthetic")


if __name__ == "__main__":
    main()
