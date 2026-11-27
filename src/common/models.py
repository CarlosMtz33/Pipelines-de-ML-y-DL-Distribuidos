"""Catálogo de modelos soportados, validación de configuraciones y job_id determinista.

Este módulo lo usan el coordinador (para validar y calcular job_id), el nodo y los
scripts de medición. Está en un solo lugar para que todos calculen el job_id igual.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

MODELS = ("linear", "mlp", "cnn")

# Alias que acepta el cliente; el coordinador siempre guarda el nombre canónico.
ALIASES = {"lineal": "linear", "regresion": "linear", "linear": "linear",
           "mlp": "mlp", "cnn": "cnn"}

# Columnas de objetos.csv que se pueden usar como variables de entrada o como objetivo.
FEATURES = {"distance", "width", "length", "height", "volume", "visibility",
            "category", "category_group", "num_lidar_pts", "num_radar_pts", "speed"}
TARGETS = {"num_lidar_pts", "num_radar_pts", "speed", "distance"}

DEFAULT_FEATURES = ["distance", "width", "length", "height", "visibility", "category"]

# Claves que NO cambian el resultado del entrenamiento: no entran al job_id.
NON_SEMANTIC_KEYS = {"checkpoint_every", "timeout_s"}


class ConfigError(ValueError):
    """Configuración inválida -> el coordinador responde INVALID_ARGUMENT."""


def _num(lo=None, hi=None, integer=False):
    def check(name, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{name} debe ser numérico")
        if integer and float(v) != int(v):
            raise ConfigError(f"{name} debe ser entero")
        v = int(v) if integer else float(v)
        if lo is not None and v < lo:
            raise ConfigError(f"{name} debe ser >= {lo}")
        if hi is not None and v > hi:
            raise ConfigError(f"{name} debe ser <= {hi}")
        return v
    return check


def _choice(*options):
    def check(name, v):
        if v not in options:
            raise ConfigError(f"{name} debe ser uno de {options}")
        return v
    return check


def _bool(name, v):
    if not isinstance(v, bool):
        raise ConfigError(f"{name} debe ser true/false")
    return v


def _features(name, v):
    if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
        raise ConfigError(f"{name} debe ser una lista de nombres de columna")
    bad = [x for x in v if x not in FEATURES]
    if bad:
        raise ConfigError(f"{name}: columnas desconocidas {bad}")
    return sorted(set(v))   # orden canónico: mismo conjunto -> mismo job_id


def _hidden(name, v):
    if not isinstance(v, list) or not v or len(v) > 6:
        raise ConfigError(f"{name} debe ser una lista de 1 a 6 enteros")
    return [_num(1, 4096, integer=True)(name, x) for x in v]


_REGRESSION_COMMON = {
    "target": ("num_lidar_pts", _choice(*sorted(TARGETS))),
    "target_transform": ("none", _choice("none", "log1p")),
    "features": (DEFAULT_FEATURES, _features),
    "normalize": (True, _bool),
}

# Esquema por modelo: clave -> (valor por defecto, validador)
SCHEMAS: dict[str, dict[str, tuple[Any, Any]]] = {
    "linear": {
        **_REGRESSION_COMMON,
        "variant": ("ridge", _choice("ols", "ridge", "lasso")),
        "alpha": (1.0, _num(1e-8, 1e6)),
        "timeout_s": (None, _num(1, 86400)),
    },
    "mlp": {
        **_REGRESSION_COMMON,
        "hidden": ([64, 32], _hidden),
        "lr": (1e-3, _num(1e-6, 1.0)),
        "epochs": (50, _num(1, 10000, integer=True)),
        "batch_size": (256, _num(1, 65536, integer=True)),
        "weight_decay": (0.0, _num(0.0, 1.0)),
        "checkpoint_every": (5, _num(1, 10000, integer=True)),
        "timeout_s": (None, _num(1, 86400)),
    },
    "cnn": {
        "filters": (16, _num(1, 256, integer=True)),
        "conv_layers": (2, _num(1, 4, integer=True)),
        "lr": (1e-3, _num(1e-6, 1.0)),
        "epochs": (8, _num(1, 1000, integer=True)),
        "img_size": (64, _choice(16, 32, 48, 64)),
        "batch_size": (64, _num(1, 4096, integer=True)),
        "weight_decay": (0.0, _num(0.0, 1.0)),
        # "balanced": cada clase pesa en la pérdida en proporción inversa a cuántos ejemplos
        # tiene en train. Sin esto, las clases raras (moto, bicicleta) se ignoran.
        "class_weight": ("balanced", _choice("balanced", "none")),
        "checkpoint_every": (1, _num(1, 1000, integer=True)),
        "timeout_s": (None, _num(1, 86400)),
    },
}


def canonical_model(model: str) -> str:
    m = ALIASES.get((model or "").strip().lower())
    if m is None:
        raise ConfigError(f"modelo '{model}' no soportado; use uno de {list(MODELS)}")
    return m


def validate_config(model: str, config: dict) -> dict:
    """Devuelve la configuración completa (con defaults) y normalizada, o lanza ConfigError."""
    model = canonical_model(model)
    if not isinstance(config, dict):
        raise ConfigError("la configuración debe ser un objeto JSON")
    schema = SCHEMAS[model]
    unknown = set(config) - set(schema)
    if unknown:
        raise ConfigError(f"claves desconocidas para {model}: {sorted(unknown)}")
    out = {}
    for key, (default, check) in schema.items():
        if key in config and config[key] is not None:
            out[key] = check(key, config[key])
        elif default is not None:
            out[key] = default
    if model in ("linear", "mlp") and out["target"] in out["features"]:
        raise ConfigError("la variable objetivo no puede estar también en features")
    return out


def expand_grid(base: dict, grid: dict | None) -> list[dict]:
    """Producto cartesiano de los hiperparámetros del grid sobre la config base.

    El orden es determinista (claves ordenadas) para que el mismo grid genere
    siempre los mismos trabajos en el mismo orden.
    """
    if not grid:
        return [dict(base)]
    if not isinstance(grid, dict) or not all(isinstance(v, list) and v for v in grid.values()):
        raise ConfigError("grid_json debe ser {param: [valores...]} con listas no vacías")
    keys = sorted(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, values)))
        configs.append(cfg)
    return configs


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_job_id(model: str, config: dict, data_version: str, seed: int) -> str:
    """job_id = hash(modelo, configuración, versión de datos, semilla).

    Mismo trabajo => mismo job_id, sin importar el orden de las claves, si se
    escribió 1 o 1.0, o qué nodo/usuario lo envió.
    """
    semantic = {k: v for k, v in config.items() if k not in NON_SEMANTIC_KEYS}
    payload = canonical_json({"model": canonical_model(model), "config": semantic,
                              "data_version": data_version, "seed": int(seed)})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_request_id(job_ids: list[str]) -> str:
    return "req-" + hashlib.sha256(",".join(sorted(job_ids)).encode()).hexdigest()[:12]


def total_epochs(model: str, config: dict) -> int:
    return int(config.get("epochs", 1)) if model in ("mlp", "cnn") else 1


def selection_metric(model: str) -> tuple[str, bool]:
    """Métrica de validación para elegir el mejor del grid: (nombre, mayor_es_mejor)."""
    return ("val_acc", True) if model == "cnn" else ("val_rmse", False)
