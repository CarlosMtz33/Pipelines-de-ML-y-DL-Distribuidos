"""Estado persistente del coordinador: la cola de trabajos en SQLite.

Todo cambio de estado se escribe en disco ANTES de responder, así que si el
coordinador se cae y se vuelve a levantar, la cola, los estados, los intentos y
los resultados siguen ahí (ver la sección "falla del coordinador" del README).

Estados de un trabajo:
    PENDING --asignar--> RUNNING --resultado--> DONE
       ^                    |
       +----reencolar-------+   (nodo caído, timeout o error; cuenta en reassignments)
                            +--> FAILED  (si se agotan los intentos)
"""
from __future__ import annotations

import sqlite3
import threading
import time

PENDING, RUNNING, DONE, FAILED = "PENDING", "RUNNING", "DONE", "FAILED"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    data_version TEXT NOT NULL,
    cost REAL NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    reassignments INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT DEFAULT '',
    lease_id TEXT DEFAULT '',
    lease_deadline REAL DEFAULT 0,
    assigned_at REAL DEFAULT 0,
    avoid_worker TEXT DEFAULT '',
    epoch INTEGER DEFAULT 0,
    total_epochs INTEGER DEFAULT 0,
    loss REAL DEFAULT 0,
    checkpoint_epoch INTEGER DEFAULT 0,
    metrics_json TEXT DEFAULT '',
    model_filename TEXT DEFAULT '',
    error TEXT DEFAULT '',
    created_at REAL NOT NULL,
    queued_at REAL NOT NULL DEFAULT 0,
    started_at REAL DEFAULT 0,
    finished_at REAL DEFAULT 0,
    train_seconds REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    name TEXT,
    model TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS request_jobs (
    request_id TEXT,
    job_id TEXT,
    PRIMARY KEY (request_id, job_id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
"""


class JobStore:
    def __init__(self, path: str):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------------- lectura
    def get(self, job_id: str):
        with self.lock:
            return self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()

    def list(self, request_id: str = "", state: str = ""):
        q, args = "SELECT j.* FROM jobs j", []
        if request_id:
            q += " JOIN request_jobs r ON r.job_id=j.job_id AND r.request_id=?"
            args.append(request_id)
        if state:
            q += " WHERE j.state=?"
            args.append(state)
        q += " ORDER BY j.seq"
        with self.lock:
            return self.db.execute(q, args).fetchall()

    def get_request(self, request_id: str):
        with self.lock:
            return self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()

    def running(self):
        with self.lock:
            return self.db.execute("SELECT * FROM jobs WHERE state=?", (RUNNING,)).fetchall()

    # ---------------------------------------------------------------- escritura
    def add_job(self, *, job_id, model, config_json, seed, data_version, cost, total_epochs):
        """Inserta el trabajo si no existe. Devuelve (fila, ya_existia)."""
        with self.lock:
            row = self.get(job_id)
            if row is not None:
                if row["state"] == FAILED:
                    # Reenviar un trabajo FALLIDO lo vuelve a intentar (sigue siendo un solo job_id).
                    self.db.execute("UPDATE jobs SET state=?, attempts=0, error='', queued_at=? WHERE job_id=?",
                                    (PENDING, time.time(), job_id))
                    return self.get(job_id), True
                return row, True
            self.db.execute(
                "INSERT INTO jobs (job_id, model, config_json, seed, data_version, cost, state,"
                " total_epochs, created_at, queued_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, model, config_json, seed, data_version, cost, PENDING, total_epochs,
                 time.time(), time.time()))
            return self.get(job_id), False

    def add_request(self, request_id, name, model, job_ids):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO requests VALUES (?,?,?,?)",
                            (request_id, name, model, time.time()))
            self.db.executemany("INSERT OR IGNORE INTO request_jobs VALUES (?,?)",
                                [(request_id, j) for j in job_ids])

    def claim_next(self, models, worker_id, lease_id, timeouts, lpt=True, avoid_grace=5.0):
        """Toma el siguiente PENDING para alguno de 'models' y lo marca RUNNING (atómico).

        lpt=True: primero los trabajos más costosos (Longest Processing Time first).
        Así las CNN arrancan al inicio y no quedan solas al final alargando el total.
        Un trabajo que falló o venció en este nodo (avoid_worker) no se le vuelve a dar
        durante avoid_grace segundos, para que lo tome otro nodo si hay alguno libre.
        """
        if not models:
            return None
        order = "cost DESC, seq ASC" if lpt else "seq ASC"
        marks = ",".join("?" * len(models))
        with self.lock:
            row = self.db.execute(
                f"SELECT * FROM jobs WHERE state=? AND model IN ({marks})"
                f" AND (avoid_worker!=? OR queued_at<?) ORDER BY {order} LIMIT 1",
                (PENDING, *models, worker_id, time.time() - avoid_grace)).fetchone()
            if row is None:
                return None
            now = time.time()
            timeout = timeouts(row)
            self.db.execute(
                "UPDATE jobs SET state=?, worker_id=?, lease_id=?, lease_deadline=?, assigned_at=?,"
                " attempts=attempts+1, started_at=CASE WHEN started_at=0 THEN ? ELSE started_at END"
                " WHERE job_id=?",
                (RUNNING, worker_id, lease_id, now + timeout, now, now, row["job_id"]))
            return self.get(row["job_id"]), timeout

    def requeue(self, job_id, reason, max_attempts, expected_lease=None):
        """Devuelve un trabajo RUNNING a la cola (o lo marca FAILED si agotó intentos)."""
        with self.lock:
            row = self.get(job_id)
            if row is None or row["state"] != RUNNING:
                return None
            if expected_lease is not None and row["lease_id"] != expected_lease:
                return None
            new_state = FAILED if row["attempts"] >= max_attempts else PENDING
            self.db.execute(
                "UPDATE jobs SET state=?, reassignments=reassignments+1, avoid_worker=worker_id,"
                " lease_id='', lease_deadline=0, error=?, queued_at=? WHERE job_id=?",
                (new_state, reason, time.time(), job_id))
            if new_state == FAILED:
                self.db.execute("UPDATE jobs SET finished_at=? WHERE job_id=?", (time.time(), job_id))
            return new_state

    def progress(self, job_id, lease_id, epoch, total, loss):
        with self.lock:
            cur = self.db.execute(
                "UPDATE jobs SET epoch=?, total_epochs=?, loss=? WHERE job_id=? AND lease_id=? AND state=?",
                (epoch, total, loss, job_id, lease_id, RUNNING))
            return cur.rowcount == 1

    def lease_is_current(self, job_id, lease_id):
        row = self.get(job_id)
        return row is not None and row["state"] == RUNNING and row["lease_id"] == lease_id

    def set_checkpoint_epoch(self, job_id, epoch):
        with self.lock:
            self.db.execute("UPDATE jobs SET checkpoint_epoch=? WHERE job_id=?", (epoch, job_id))

    def finish(self, job_id, worker_id, metrics_json, model_filename, train_seconds):
        """Marca DONE solo si aún no lo estaba. Devuelve False si ya había resultado."""
        with self.lock:
            cur = self.db.execute(
                "UPDATE jobs SET state=?, worker_id=?, metrics_json=?, model_filename=?,"
                " train_seconds=?, finished_at=?, lease_id='', lease_deadline=0, error=''"
                " WHERE job_id=? AND state!=?",
                (DONE, worker_id, metrics_json, model_filename, train_seconds, time.time(), job_id, DONE))
            return cur.rowcount == 1
