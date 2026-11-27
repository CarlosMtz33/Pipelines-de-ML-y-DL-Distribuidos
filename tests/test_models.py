"""Pruebas de la fase 3: validación de configuraciones y job_id determinista.

  python tests/test_models.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.models import ConfigError, compute_job_id, expand_grid, validate_config  # noqa: E402


def jid(model, cfg, version="v1", seed=42):
    return compute_job_id(model, validate_config(model, cfg), version, seed)


# 1. Escribir 1 o 1.0 da el mismo trabajo
assert jid("linear", {"alpha": 1}) == jid("linear", {"alpha": 1.0})
print("ok 1: {'alpha': 1} y {'alpha': 1.0} dan el mismo job_id")

# 2. Otra semilla u otra versión de datos -> otro trabajo
assert jid("linear", {"alpha": 1}) != jid("linear", {"alpha": 1}, seed=7)
assert jid("linear", {"alpha": 1}) != jid("linear", {"alpha": 1}, version="v2")
print("ok 2: otra semilla u otra versión de datos da otro job_id")

# 3. checkpoint_every no cambia el resultado, así que no cambia el job_id
assert jid("mlp", {"checkpoint_every": 5}) == jid("mlp", {"checkpoint_every": 10})
print("ok 3: checkpoint_every no cambia el job_id")

# 4. Grid search: 2 variantes x 4 alphas = 8 configuraciones
grid = expand_grid({}, {"variant": ["ridge", "lasso"], "alpha": [0.01, 0.1, 1, 10]})
assert len(grid) == 8
print("ok 4: el grid de 2 x 4 da 8 configuraciones")

# 5. Modelo inválido y clave mal escrita se rechazan
for model, cfg in [("svm", {}), ("mlp", {"hiden": [32]})]:
    try:
        validate_config(model, cfg)
    except ConfigError as e:
        print(f"ok 5: {model} {cfg} rechazado -> {e}")
    else:
        raise AssertionError(f"{model} {cfg} debió rechazarse")

print("\nTodas las pruebas pasaron")
