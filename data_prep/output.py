"""Escritura de los datos preparados en el formato que esperan los nodos.

Archivos que se generan en el directorio de salida:
  objetos.csv        una fila por objeto anotado (tabla para regresión lineal y MLP)
  recortes.npy       (N, 64, 64, 3) uint8, recorte de cámara de cada objeto visible
  recortes_meta.csv  una fila por recorte: token, escena, split, grupo y etiqueta numérica
  clases.json        nombres de las clases de la CNN, en el orden de las etiquetas
  particion.json     qué escenas van a train/val/test
  VERSION            hash del contenido: identifica los datos y entra al job_id
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

OBJETOS_COLS = ["annotation_token", "sample_token", "scene", "split", "category_full",
                "category", "category_group", "width", "length", "height", "volume",
                "distance", "visibility", "num_lidar_pts", "num_radar_pts",
                "vel_x", "vel_y", "speed"]


def category_group(name: str, cfg: dict) -> str:
    for prefix, group in cfg["grupos_categoria"]:
        if name.startswith(prefix):
            return group
    return cfg["grupo_por_defecto"]


def category_2levels(name: str) -> str:
    return ".".join(name.split(".")[:2])


def write_prepared(out_dir: str, objetos: pd.DataFrame, recortes: np.ndarray,
                   meta: pd.DataFrame, cfg: dict, prefix: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    clases = cfg["clases_cnn"]
    # Solo entran a la CNN los recortes de las clases elegidas; los demás objetos
    # (camiones, conos, barreras...) siguen en objetos.csv para la regresión.
    meta = meta[meta["category_group"].isin(clases)].copy()
    meta["label"] = meta["category_group"].map({c: i for i, c in enumerate(clases)}).astype(int)
    recortes = recortes[meta.index.to_numpy()]
    meta = meta.reset_index(drop=True)

    objetos = objetos[OBJETOS_COLS].sort_values(["scene", "sample_token", "annotation_token"])
    order = meta.sort_values(["scene", "annotation_token"]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    recortes = np.ascontiguousarray(recortes[order]).astype(np.uint8)

    files = {
        "objetos.csv": objetos.to_csv(index=False, float_format="%.6f").encode(),
        "recortes_meta.csv": meta.to_csv(index=False).encode(),
        "clases.json": json.dumps(clases).encode(),
        "particion.json": json.dumps(cfg["particion"], indent=2).encode(),
    }
    h = hashlib.sha256()
    for name in sorted(files):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(files[name])
        h.update(name.encode() + files[name])
    np.save(os.path.join(out_dir, "recortes.npy"), recortes)
    h.update(recortes.tobytes())
    version = f"{prefix}-{h.hexdigest()[:12]}"
    with open(os.path.join(out_dir, "VERSION"), "w") as f:
        f.write(version + "\n")

    print(f"Datos preparados en {out_dir}  VERSION={version}")
    print(f"  objetos.csv: {len(objetos)} filas")
    print(objetos.groupby("split").size().to_string())
    print(f"  recortes.npy: {recortes.shape}")
    print(meta.groupby(["split", "category_group"]).size().unstack(fill_value=0).to_string())
    return version
