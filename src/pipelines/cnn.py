"""Plantilla de pipeline: CNN para clasificar recortes de cámara en su categoría agrupada.

Datos: recortes.npy (N, 64, 64, 3) uint8 y recortes_meta.csv (split y etiqueta).
Métricas: exactitud y matriz de confusión (más exactitud balanceada, porque las
clases están desbalanceadas: hay muchos más carros que bicicletas o motos).
"""
from __future__ import annotations

import threading

import numpy as np

from .common import (build_seeded, Dataset, JobContext, PipelineResult, run_torch_training,
                     torch_model_bytes, torch_setup)

_resize_lock = threading.Lock()
_resize_cache: dict = {}


def resized_images(data: Dataset, size: int):
    """Recortes como tensor uint8 (N, 3, size, size). Se calcula una vez por nodo y tamaño.

    Se guardan en uint8 (no float) para ahorrar memoria: se convierten a float
    lote por lote durante el entrenamiento.
    """
    import torch
    import torch.nn.functional as F

    key = (data.data_dir, data.version, size)
    with _resize_lock:
        if key not in _resize_cache:
            imgs = torch.from_numpy(np.array(data.recortes, copy=True)).permute(0, 3, 1, 2)
            if size != imgs.shape[-1]:
                out = []
                for i in range(0, len(imgs), 2048):   # por bloques para no gastar memoria
                    chunk = imgs[i:i + 2048].float()
                    out.append(F.interpolate(chunk, size=(size, size), mode="area")
                               .round().clamp(0, 255).to(torch.uint8))
                imgs = torch.cat(out)
            _resize_cache[key] = imgs.contiguous()
        return _resize_cache[key]


def build_cnn(n_classes: int, filters: int, conv_layers: int, img_size: int):
    import torch.nn as nn
    layers, c_in, c_out, size = [], 3, filters, img_size
    for _ in range(conv_layers):
        layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        c_in, c_out, size = c_out, c_out * 2, size // 2
    layers += [nn.Flatten(), nn.Linear(c_in * size * size, 64), nn.ReLU(), nn.Linear(64, n_classes)]
    return nn.Sequential(*layers)


def confusion(y_true, y_pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def normalized(cm):
    """Matriz de confusión normalizada por fila (clase real): la diagonal ideal es 1."""
    sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, sums, out=np.zeros(cm.shape, dtype=float), where=sums > 0).round(4)


def balanced_accuracy(cm):
    rec = [cm[i, i] / cm[i].sum() for i in range(len(cm)) if cm[i].sum() > 0]
    return float(np.mean(rec)) if rec else 0.0


def run(config: dict, seed: int, data: Dataset, ctx: JobContext) -> PipelineResult:
    import torch

    torch_setup(seed)
    size = config["img_size"]
    imgs = resized_images(data, size)
    labels = torch.tensor(data.recortes_meta["label"].to_numpy(), dtype=torch.long)
    split = data.recortes_meta["split"].to_numpy()
    idx = {s: torch.tensor(np.flatnonzero(split == s)) for s in ("train", "val", "test")}
    n_classes = len(data.clases)

    # Media y desviación por canal, SOLO con train.
    tr = imgs[idx["train"]].float() / 255.0
    mean = tr.mean(dim=(0, 2, 3), keepdim=True)
    std = tr.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    del tr

    def x_of(ix):
        return (imgs[ix].float() / 255.0 - mean) / std

    model = build_seeded(seed, lambda: build_cnn(n_classes, config["filters"], config["conv_layers"], size))
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["class_weight"] == "balanced":
        counts = torch.bincount(labels[idx["train"]], minlength=n_classes).float().clamp_min(1)
        weights = counts.sum() / (n_classes * counts)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    def predict(ix):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(ix), 512):
                preds.append(model(x_of(ix[i:i + 512])).argmax(1))
        return torch.cat(preds).numpy()

    def evaluate(split_name):
        ix = idx[split_name]
        cm = confusion(labels[ix].numpy(), predict(ix), n_classes)
        return float(np.trace(cm) / cm.sum()), cm

    def on_epoch_end(epoch):
        return {"val_acc": evaluate("val")[0]}

    train_ix = idx["train"]
    history, resumed_from = run_torch_training(
        model=model, optimizer=opt, loss_fn=loss_fn, n_train=len(train_ix),
        batch_size=config["batch_size"],
        get_batch=lambda b: (x_of(train_ix[b]), labels[train_ix[b]]),
        epochs=config["epochs"], seed=seed, ctx=ctx,
        checkpoint_every=config["checkpoint_every"], on_epoch_end=on_epoch_end)

    tr_acc, _ = evaluate("train")
    va_acc, va_cm = evaluate("val")
    te_acc, te_cm = evaluate("test")
    metrics = {"train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc,
               "val_balanced_acc": balanced_accuracy(va_cm),
               "test_balanced_acc": balanced_accuracy(te_cm),
               "test_confusion": te_cm.tolist(), "val_confusion": va_cm.tolist(),
               "test_confusion_norm": normalized(te_cm).tolist(),
               "val_confusion_norm": normalized(va_cm).tolist(),
               "classes": data.clases, "final_train_loss": history[-1]["loss"],
               "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
               "n_test": int(len(idx["test"]))}
    model_bytes = torch_model_bytes({"state_dict": model.state_dict(), "config": config,
                                     "classes": data.clases, "mean": mean, "std": std})
    return PipelineResult(metrics, model_bytes, "model.pt",
                          extra={"history": history, "resumed_from_epoch": resumed_from})
