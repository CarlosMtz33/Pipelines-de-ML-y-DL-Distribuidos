"""Plantilla de pipeline: regresión lineal (línea base) con scikit-learn.

Tarea: predecir config["target"] (por defecto num_lidar_pts) a partir de
distancia, tamaño, categoría y visibilidad. Métricas: RMSE y R².
"""
from __future__ import annotations

import pickle

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (Dataset, JobContext, PipelineResult, inverse_target,
                     regression_arrays, regression_metrics, set_seed, transform_target)


def run(config: dict, seed: int, data: Dataset, ctx: JobContext) -> PipelineResult:
    set_seed(seed)
    parts, feature_names = regression_arrays(data, config)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    how = config["target_transform"]

    variant = config["variant"]
    if variant == "ridge":
        reg = Ridge(alpha=config["alpha"], random_state=seed)
    elif variant == "lasso":
        reg = Lasso(alpha=config["alpha"], random_state=seed, max_iter=20000)
    else:
        reg = LinearRegression()
    steps = [StandardScaler()] if config["normalize"] else []
    pipe = make_pipeline(*steps, reg)

    ctx.check_cancel()
    pipe.fit(Xtr, transform_target(ytr, how))

    def predict(X):
        return inverse_target(pipe.predict(X), how)

    m_tr, m_va, m_te = (regression_metrics(y, predict(X))
                        for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte)))
    ctx.progress(1, 1, m_tr["rmse"] ** 2, val_rmse=m_va["rmse"])

    metrics = {**{f"{part}_{k}": m[k] for part, m in (("train", m_tr), ("val", m_va), ("test", m_te))
                  for k in ("mse", "rmse", "mae", "r2")},
               "n_train": m_tr["n"], "n_val": m_va["n"], "n_test": m_te["n"]}
    model_bytes = pickle.dumps({"pipeline": pipe, "features": feature_names, "config": config})
    return PipelineResult(metrics, model_bytes, "model.pkl")
