"""Piezas compartidas por las tres plantillas de pipeline.

Interfaz común de las plantillas (linear.py, mlp.py, cnn.py):

    run(config: dict, seed: int, data: Dataset, ctx: JobContext) -> PipelineResult

- config: configuración ya validada (con defaults), ver common/models.py
- seed:   semilla; con la misma semilla el resultado es idéntico
- data:   datos preparados (objetos.csv, recortes.npy) cargados una vez por nodo
- ctx:    canal hacia afuera: avance, checkpoints y cancelación. En el nodo lo
          implementa el worker (manda todo al coordinador por gRPC); en la
          línea base sin sistema es un contexto local que no hace nada.
"""
from __future__ import annotations

import io
import json
import os
import random
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd


class JobCancelled(Exception):
    """El coordinador pidió abandonar el trabajo (fue reasignado o ya terminó en otro nodo)."""


class JobContext:
    """Contexto 'nulo': lo usa la ejecución de referencia sin sistema distribuido."""

    checkpoint: bytes | None = None    # checkpoint con el que se reanuda (si hay)
    checkpoint_epoch: int = 0

    def progress(self, epoch: int, total: int, loss: float, **extra) -> None:
        pass

    def save_checkpoint(self, epoch: int, data: bytes) -> None:
        pass

    def check_cancel(self) -> None:
        pass


@dataclass
class PipelineResult:
    metrics: dict
    model_bytes: bytes
    model_filename: str
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ datos

@dataclass
class Dataset:
    data_dir: str
    version: str
    objetos: pd.DataFrame
    recortes: np.ndarray          # (N, 64, 64, 3) uint8, abierto con mmap
    recortes_meta: pd.DataFrame   # una fila por recorte: split, label, ...
    clases: list[str]


_data_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(data_dir: str) -> Dataset:
    with open(os.path.join(data_dir, "VERSION")) as f:
        version = f.read().strip()
    objetos = pd.read_csv(os.path.join(data_dir, "objetos.csv"))
    recortes = np.load(os.path.join(data_dir, "recortes.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(data_dir, "recortes_meta.csv"))
    with open(os.path.join(data_dir, "clases.json")) as f:
        clases = json.load(f)
    if len(meta) != recortes.shape[0]:
        raise RuntimeError("recortes.npy y recortes_meta.csv no tienen el mismo número de filas")
    return Dataset(data_dir, version, objetos, recortes, meta, clases)


def load_dataset(data_dir: str) -> Dataset:
    with _data_lock:
        return _load(os.path.abspath(data_dir))


# ------------------------------------------------------------------ reproducibilidad

def set_seed(seed: int) -> None:
    """Semillas globales de random/numpy (los modelos de sklearn reciben random_state aparte).

    La de PyTorch NO se fija aquí: se fija dentro de build_seeded(), con candado.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))


# ------------------------------------------------------------------ regresión tabular

def regression_arrays(data: Dataset, config: dict):
    """Arma X, y por partición a partir de objetos.csv según la config.

    La partición viene de la columna 'split' (por escena), que fijó la preparación
    de datos; aquí no se vuelve a partir nada al azar.
    """
    df = data.objetos
    target = config["target"]
    feats = config["features"]
    needed = [c for c in feats if c not in ("category", "category_group")] + [target]
    df = df.dropna(subset=needed)

    cols, names = [], []
    for f in feats:
        if f in ("category", "category_group"):
            # Las categorías se toman de TODO el archivo (no solo de train) para que
            # las columnas one-hot sean idénticas en cualquier nodo y partición.
            levels = sorted(data.objetos[f].dropna().unique())
            for lv in levels:
                cols.append((df[f] == lv).astype(np.float64).to_numpy())
                names.append(f"{f}={lv}")
        else:
            cols.append(df[f].astype(np.float64).to_numpy())
            names.append(f)
    X = np.column_stack(cols)
    y = df[target].astype(np.float64).to_numpy()
    split = df["split"].to_numpy()
    parts = {s: (X[split == s], y[split == s]) for s in ("train", "val", "test")}
    for s, (xs, _) in parts.items():
        if len(xs) == 0:
            raise RuntimeError(f"la partición '{s}' quedó vacía para target={target}")
    return parts, names


def transform_target(y, how):
    return np.log1p(np.clip(y, 0, None)) if how == "log1p" else y


def inverse_target(y, how):
    return np.expm1(y) if how == "log1p" else y


def regression_metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Rangos: MSE y RMSE en [0, inf) (0 = perfecto); MAE en [0, inf); R2 en (-inf, 1] (1 = perfecto,
    # 0 = igual que predecir siempre el promedio, negativo = peor que eso).
    return {"mse": rmse ** 2, "rmse": rmse, "r2": float(r2), "mae": float(np.mean(np.abs(err))),
            "n": int(len(y_true))}


# ------------------------------------------------------------------ entrenamiento PyTorch

_rng_lock = threading.Lock()


def torch_setup(seed: int):
    import torch
    torch.use_deterministic_algorithms(True)


def build_seeded(seed: int, builder):
    """Construye el modelo con pesos iniciales que dependen SOLO de la semilla.

    El generador aleatorio global de PyTorch es compartido por todos los hilos del
    proceso, y un nodo corre varios trabajos a la vez en hilos. Sin este candado, la
    inicialización de un trabajo dependería de qué otro trabajo corría al lado, y
    repetir el mismo trabajo daría otras métricas (rompe la idempotencia).
    """
    import torch
    with _rng_lock:
        torch.manual_seed(seed)
        return builder()


def run_torch_training(*, model, optimizer, loss_fn, n_train: int, batch_size: int,
                       get_batch, epochs: int, seed: int, ctx: JobContext,
                       checkpoint_every: int, on_epoch_end=None):
    """Bucle de entrenamiento con checkpoints reanudables.

    Reproducibilidad al reanudar: al inicio de CADA época se resiembra el generador
    del orden de los lotes con (seed, época). Así el orden de los lotes de la época k no
    depende de cuántas épocas corrió este proceso, y reanudar desde un
    checkpoint da exactamente el mismo modelo que correr de corrido.
    """
    import torch

    start_epoch, history = 0, []
    if ctx.checkpoint:
        state = torch.load(io.BytesIO(ctx.checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, history = state["epoch"], state["history"]

    for epoch in range(start_epoch, epochs):
        ctx.check_cancel()
        # Generador PROPIO del trabajo (no el global), resembrado en cada época.
        g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        perm = torch.randperm(n_train, generator=g)
        model.train()
        total, count = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = get_batch(idx)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
            count += len(idx)
        epoch_loss = total / max(count, 1)
        extra = on_epoch_end(epoch) if on_epoch_end else {}
        history.append({"epoch": epoch + 1, "loss": epoch_loss, **extra})
        ctx.progress(epoch + 1, epochs, epoch_loss, **extra)

        if (epoch + 1) % checkpoint_every == 0 and epoch + 1 < epochs:
            buf = io.BytesIO()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1, "history": history}, buf)
            ctx.save_checkpoint(epoch + 1, buf.getvalue())
    return history, start_epoch


def torch_model_bytes(obj: dict) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()
