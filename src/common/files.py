"""Escritura atómica de archivos.

Se escribe a un archivo temporal en el MISMO directorio y luego se renombra con
os.replace, que es atómico en el mismo sistema de archivos. Quien lea el archivo
ve la versión anterior completa o la nueva completa, nunca una a medias.
"""
from __future__ import annotations

import json
import os
import tempfile


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True).encode())


def read_data_version(data_dir: str) -> str:
    with open(os.path.join(data_dir, "VERSION")) as f:
        return f.read().strip()
