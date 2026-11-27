"""Registro de plantillas: el nodo elige la plantilla según el tipo de modelo."""
from . import cnn, linear, mlp

PIPELINES = {"linear": linear.run, "mlp": mlp.run, "cnn": cnn.run}
