"""Plantilla de pipeline: MLP (perceptrón multicapa) con PyTorch, solo CPU.

Misma tarea que la regresión lineal, para comparar contra ella. Guarda checkpoints
cada config["checkpoint_every"] épocas para poder reanudar si el nodo cae.
"""
from __future__ import annotations

import numpy as np

from .common import (build_seeded, Dataset, JobContext, PipelineResult, inverse_target,
                     regression_arrays, regression_metrics, run_torch_training,
                     torch_model_bytes, torch_setup, transform_target)


def build_mlp(n_in: int, hidden: list[int]):
    import torch.nn as nn
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def run(config: dict, seed: int, data: Dataset, ctx: JobContext) -> PipelineResult:
    import torch

    torch_setup(seed)
    parts, feature_names = regression_arrays(data, config)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    how = config["target_transform"]

    # Normalización de X ajustada SOLO con train (si no, se filtra info de val/test).
    if config["normalize"]:
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
    else:
        mu, sd = np.zeros(Xtr.shape[1]), np.ones(Xtr.shape[1])
    # La salida se estandariza siempre: ayuda a que la red converja.
    ttr = transform_target(ytr, how)
    y_mu, y_sd = float(ttr.mean()), float(ttr.std() or 1.0)

    def to_x(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32)

    xtr = to_x(Xtr)
    ytr_t = torch.tensor((ttr - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)

    model = build_seeded(seed, lambda: build_mlp(Xtr.shape[1], config["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    def predict(X):
        model.eval()
        with torch.no_grad():
            out = model(to_x(X)).squeeze(1).numpy().astype(np.float64)
        return inverse_target(out * y_sd + y_mu, how)

    def on_epoch_end(epoch):
        return {"val_rmse": regression_metrics(yva, predict(Xva))["rmse"]}

    history, resumed_from = run_torch_training(
        model=model, optimizer=opt, loss_fn=loss_fn, n_train=len(xtr),
        batch_size=config["batch_size"], get_batch=lambda idx: (xtr[idx], ytr_t[idx]),
        epochs=config["epochs"], seed=seed, ctx=ctx,
        checkpoint_every=config["checkpoint_every"], on_epoch_end=on_epoch_end)

    m_tr, m_va, m_te = (regression_metrics(y, predict(X))
                        for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte)))
    metrics = {**{f"{part}_{k}": m[k] for part, m in (("train", m_tr), ("val", m_va), ("test", m_te))
                  for k in ("mse", "rmse", "mae", "r2")}, "final_train_loss": history[-1]["loss"],
               "n_train": m_tr["n"], "n_val": m_va["n"], "n_test": m_te["n"]}
    model_bytes = torch_model_bytes({
        "state_dict": model.state_dict(), "config": config, "features": feature_names,
        "x_mean": mu.tolist(), "x_std": sd.tolist(), "y_mean": y_mu, "y_std": y_sd})
    return PipelineResult(metrics, model_bytes, "model.pt",
                          extra={"history": history, "resumed_from_epoch": resumed_from})
