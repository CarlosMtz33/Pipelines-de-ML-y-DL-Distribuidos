"""Prueba: el mismo trabajo da las mismas métricas solo o con otros trabajos corriendo a la vez.

Simula lo que pasa en un nodo (varios trabajos en hilos del mismo proceso) y lo
compara con la ejecución secuencial. Si falla, la idempotencia no se puede comprobar.

  PYTHONPATH=src python tests/test_determinismo.py data/synthetic
"""
import sys
import threading

import torch

from common.models import validate_config
from pipelines import PIPELINES
from pipelines.common import JobContext, load_dataset

torch.set_num_threads(1)
data = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
jobs = [("mlp", {"epochs": 6, "hidden": [32]}), ("mlp", {"epochs": 6, "hidden": [64, 32]}),
        ("cnn", {"epochs": 2, "img_size": 16, "filters": 8}), ("cnn", {"epochs": 2, "img_size": 16, "filters": 4}),
        ("linear", {"variant": "lasso", "alpha": 0.1})]
jobs = [(m, validate_config(m, c)) for m, c in jobs]

seq = [PIPELINES[m](c, 42, data, JobContext()).metrics for m, c in jobs]
par = [None] * len(jobs)


def run(i):
    m, c = jobs[i]
    par[i] = PIPELINES[m](c, 42, data, JobContext()).metrics


threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
[t.start() for t in threads]
[t.join() for t in threads]

bad = 0
for (m, _), a, b in zip(jobs, seq, par):
    key = "test_acc" if m == "cnn" else "test_rmse"
    same = a[key] == b[key]
    bad += not same
    print(f"{m:6s} {key}: secuencial={a[key]:.6f} en paralelo={b[key]:.6f} {'OK' if same else 'DISTINTO'}")
sys.exit(1 if bad else 0)
