"""Retry con backoff exponencial y jitter para llamadas gRPC.

Solo se reintentan errores pasajeros (UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED).
Un INVALID_ARGUMENT, por ejemplo, no se arregla reintentando.

Reintentar es seguro porque todas las operaciones son idempotentes: el job_id es
determinista y el coordinador guarda un solo resultado por trabajo.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

import grpc

T = TypeVar("T")
log = logging.getLogger("retry")

TRANSIENT = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED,
             grpc.StatusCode.RESOURCE_EXHAUSTED}


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """'Full jitter': espera aleatoria entre 0 y min(cap, base * 2^intento).

    El jitter evita que muchos nodos reintenten todos al mismo tiempo cuando el
    coordinador vuelve (efecto manada).
    """
    return random.uniform(0, min(cap, base * (2 ** attempt)))


def call_with_retry(fn: Callable[[], T], *, what: str, max_attempts: int = 6,
                    base: float = 0.5, cap: float = 10.0) -> T:
    """Llama fn(); si falla por causa pasajera, espera con backoff y reintenta.

    fn debe hacer la llamada gRPC con su propio timeout (parámetro timeout=...).
    max_attempts <= 0 significa reintentar indefinidamente.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            attempt += 1
            if code not in TRANSIENT or (max_attempts > 0 and attempt >= max_attempts):
                raise
            delay = backoff_delay(attempt - 1, base, cap)
            log.warning("%s falló (%s), intento %d, reintento en %.2fs",
                        what, code.name if code else e, attempt, delay)
            time.sleep(delay)
