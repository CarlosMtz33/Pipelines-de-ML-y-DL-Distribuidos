"""Entrena un modelo de cada tipo con los datos preparados y muestra sus métricas.

Sirve para comprobar la fase 4 sin coordinador ni nodos (todo en un solo proceso).

  python tests/probar_modelos.py data/prepared
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.models import validate_config  # noqa: E402
from pipelines import PIPELINES  # noqa: E402
from pipelines.common import JobContext, load_dataset  # noqa: E402

torch.set_num_threads(2)
data = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "data/prepared")
print(f"Datos {data.version}: {len(data.objetos)} objetos, {len(data.recortes_meta)} recortes, clases {data.clases}\n")

trabajos = [
    ("linear", {"variant": "ridge", "alpha": 1, "target_transform": "log1p"}),
    ("mlp", {"hidden": [64, 32], "lr": 0.001, "epochs": 30, "target_transform": "log1p"}),
    ("cnn", {"filters": 8, "img_size": 32, "epochs": 6}),
]
for model, cfg in trabajos:
    t = time.time()
    m = PIPELINES[model](validate_config(model, cfg), 42, data, JobContext()).metrics
    dt = time.time() - t
    print(f"{model} {json.dumps(cfg)}  ({dt:.1f} s)")
    if model == "cnn":
        print(f"   exactitud: val={m['val_acc']:.3f} test={m['test_acc']:.3f}  balanceada test={m['test_balanced_acc']:.3f}")
        print("   matriz de confusión normalizada (test), filas = clase real:")
        print("   " + " ".join(f"{c[:6]:>7s}" for c in m["classes"]))
        for c, fila in zip(m["classes"], np.array(m["test_confusion_norm"])):
            print(f"   {' '.join(f'{v:7.2f}' for v in fila)}   <- {c}")
    else:
        print(f"   test: R2={m['test_r2']:.3f}  MSE={m['test_mse']:.1f}  RMSE={m['test_rmse']:.2f}  MAE={m['test_mae']:.2f}")
    print()
