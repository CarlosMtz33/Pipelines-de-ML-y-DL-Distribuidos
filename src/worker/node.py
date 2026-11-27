"""Nodo (worker): pide trabajos al coordinador, corre el pipeline y reporta el resultado.

Patrones que viven aquí:
  - Heartbeat: un hilo late cada interval_ms; el coordinador responde qué trabajos cancelar.
  - Consumidores competidores: el nodo PIDE trabajo solo cuando tiene un espacio libre.
  - Bulkhead: espacios separados por tipo de modelo (p. ej. 4 lineales, 2 MLP, 1 CNN)
    y un máximo total, para que una CNN no tire al nodo.
  - Retry con backoff: todas las llamadas al coordinador (common/retry.py).
  - Timeout: cada llamada gRPC lleva su plazo (timeout=...).
  - Escritura atómica: el resultado local se escribe a temporal y se renombra.

Variables de entorno:
  COORDINATOR_ADDR=coordinator:50051  WORKER_ID=<hostname>  DATA_DIR=/data
  RESULTS_DIR=/results  SLOTS=linear=4,mlp=2,cnn=1  MAX_JOBS=4  TORCH_THREADS=2
  SLOW_FACTOR=0 (inyección de falla: segundos extra de espera por época, para simular un nodo lento)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import tempfile
import threading
import time
import traceback

import grpc

import trainer_pb2 as pb
import trainer_pb2_grpc as pb_grpc
from common import models as M
from common.files import atomic_write_bytes, atomic_write_json, read_data_version
from common.retry import backoff_delay, call_with_retry
from pipelines import PIPELINES
from pipelines.common import JobCancelled, JobContext, load_dataset

log = logging.getLogger("worker")
# Archivo que el health check de Docker revisa. En el contenedor queda en /tmp/worker_ready;
# en Windows (pruebas locales) queda en la carpeta temporal del usuario.
READY_FILE = os.environ.get("READY_FILE", os.path.join(tempfile.gettempdir(), "worker_ready"))


def parse_slots(text: str) -> dict[str, int]:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=")
            out[M.canonical_model(k.strip())] = int(v)
    return out


class RemoteContext(JobContext):
    """Contexto que conecta el pipeline con el coordinador (avance, checkpoints, cancelación)."""

    def __init__(self, node: "Node", a: pb.Assignment):
        self.node, self.a = node, a
        self.cancelled = threading.Event()
        self.checkpoint = a.checkpoint or None
        self.checkpoint_epoch = a.checkpoint_epoch
        self.slow = float(os.environ.get("SLOW_FACTOR", 0))

    def check_cancel(self):
        if self.cancelled.is_set():
            raise JobCancelled()

    def progress(self, epoch, total, loss, **extra):
        if self.slow:
            time.sleep(self.slow)   # nodo lento simulado
        try:   # el avance es "mejor esfuerzo": si falla no se reintenta
            ack = self.node.stub.ReportProgress(pb.ProgressReport(
                job_id=self.a.job_id, lease_id=self.a.lease_id, worker_id=self.node.worker_id,
                epoch=epoch, total_epochs=total, loss=loss, extra_json=json.dumps(extra)), timeout=5)
            if ack.cancel:
                self.cancelled.set()
        except grpc.RpcError:
            pass
        self.check_cancel()

    def save_checkpoint(self, epoch, data):
        try:
            ack = call_with_retry(lambda: self.node.stub.SaveCheckpoint(pb.CheckpointUpload(
                job_id=self.a.job_id, lease_id=self.a.lease_id, worker_id=self.node.worker_id,
                epoch=epoch, data=data), timeout=30), what="SaveCheckpoint", max_attempts=4)
            if ack.cancel:
                self.cancelled.set()
        except grpc.RpcError as e:
            log.warning("No se pudo guardar checkpoint de %s: %s", self.a.job_id, e.code())


class Node:
    def __init__(self):
        self.worker_id = os.environ.get("WORKER_ID") or socket.gethostname()
        self.addr = os.environ.get("COORDINATOR_ADDR", "coordinator:50051")
        self.data_dir = os.environ.get("DATA_DIR", "/data")
        self.results_dir = os.environ.get("RESULTS_DIR", "/results")
        self.slots = parse_slots(os.environ.get("SLOTS", "linear=4,mlp=2,cnn=1"))
        self.max_jobs = int(os.environ.get("MAX_JOBS", 4))
        self.data_version = read_data_version(self.data_dir)

        opts = [("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                # Si el coordinador se cae, reintentar la conexión pronto (máx. cada 3 s).
                ("grpc.initial_reconnect_backoff_ms", 500),
                ("grpc.max_reconnect_backoff_ms", 3000)]
        self.channel = grpc.insecure_channel(self.addr, options=opts)
        self.stub = pb_grpc.CoordinatorStub(self.channel)

        self.lock = threading.Lock()
        self.running: dict[str, tuple[str, RemoteContext]] = {}   # job_id -> (modelo, ctx)
        self.ready = False
        self.accepted = True
        self.hb_interval = 2.0
        self.stop = threading.Event()

    # ------------------------------------------------------------ arranque
    def warm_up(self):
        """Carga los datos una vez antes de declararse listo (health check)."""
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))
        t = time.time()
        self.data = load_dataset(self.data_dir)
        log.info("Datos %s cargados en %.1fs (%d objetos, %d recortes)", self.data.version,
                 time.time() - t, len(self.data.objetos), len(self.data.recortes_meta))
        self.ready = True

    # ------------------------------------------------------------ heartbeat
    def heartbeat_loop(self):
        failures = 0
        while not self.stop.is_set():
            with self.lock:
                running = list(self.running)
            try:
                reply = self.stub.Heartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id, hostname=socket.gethostname(),
                    data_version=self.data_version, ready=self.ready,
                    running_job_ids=running, slots=self.slots), timeout=3)
                failures = 0
                self.hb_interval = max(reply.interval_ms / 1000.0, 0.2)
                if not reply.accepted and self.accepted:
                    log.error("El coordinador rechazó este nodo: %s", reply.message)
                self.accepted = reply.accepted
                for job_id in reply.cancel_job_ids:
                    self._cancel(job_id, "el coordinador lo reasignó o ya terminó")
                if self.ready:
                    try:
                        with open(READY_FILE, "w") as f:
                            f.write(str(time.time()))
                    except OSError:
                        pass   # sin archivo solo falla el health check de Docker, no el nodo
                wait = self.hb_interval
            except grpc.RpcError as e:
                failures += 1
                wait = min(self.hb_interval, backoff_delay(failures, 0.5, self.hb_interval)) or 0.2
                if failures in (1, 5) or failures % 30 == 0:
                    log.warning("Heartbeat falló (%s: %s), intento %d", e.code().name, e.details(), failures)
            self.stop.wait(wait)

    def _cancel(self, job_id, why):
        with self.lock:
            entry = self.running.get(job_id)
        if entry and not entry[1].cancelled.is_set():
            log.warning("Cancelando %s: %s", job_id, why)
            entry[1].cancelled.set()

    # ------------------------------------------------------------ bulkhead
    def free_models(self) -> list[str]:
        with self.lock:
            if len(self.running) >= self.max_jobs:
                return []
            busy = {}
            for model, _ in self.running.values():
                busy[model] = busy.get(model, 0) + 1
        return [m for m, cap in self.slots.items() if busy.get(m, 0) < cap]

    # ------------------------------------------------------------ consumir trabajos
    def pull_loop(self):
        idle = 0
        while not self.stop.is_set():
            free = self.free_models() if (self.ready and self.accepted) else []
            if not free:
                self.stop.wait(0.2)
                continue
            try:
                reply = self.stub.PullJob(pb.PullRequest(worker_id=self.worker_id,
                                                         accept_models=free), timeout=5)
            except grpc.RpcError as e:
                idle += 1
                if e.code() != grpc.StatusCode.FAILED_PRECONDITION:
                    log.warning("PullJob falló (%s)", e.code().name)
                self.stop.wait(backoff_delay(min(idle, 5), 0.3, 5.0))
                continue
            if not reply.has_job:
                idle += 1
                self.stop.wait(min(0.2 * idle, 1.0))   # cola vacía: espera un poco
                continue
            idle = 0
            a = reply.job
            ctx = RemoteContext(self, a)
            with self.lock:
                self.running[a.job_id] = (a.model, ctx)
            threading.Thread(target=self.run_job, args=(a, ctx), daemon=True,
                             name=f"job-{a.job_id}").start()

    def run_job(self, a: pb.Assignment, ctx: RemoteContext):
        resume = f" (reanuda desde época {a.checkpoint_epoch})" if a.checkpoint else ""
        log.info("Inicia %s %s intento %d%s", a.model, a.job_id, a.attempt, resume)
        t0 = time.time()
        try:
            if a.data_version != self.data_version:
                raise RuntimeError(f"datos distintos: trabajo={a.data_version} nodo={self.data_version}")
            config = json.loads(a.config_json)
            result = PIPELINES[a.model](config, a.seed, self.data, ctx)
            elapsed = time.time() - t0
            metrics = dict(result.metrics)
            metrics["resumed_from_epoch"] = result.extra.get("resumed_from_epoch", 0)
            # Copia local atómica (por si el coordinador no está disponible en este momento).
            local = os.path.join(self.results_dir, a.job_id)
            atomic_write_bytes(os.path.join(local, result.model_filename), result.model_bytes)
            atomic_write_json(os.path.join(local, "metrics.json"),
                              {"job_id": a.job_id, "metrics": metrics, "train_seconds": elapsed,
                               "history": result.extra.get("history")})
            msg = pb.JobResult(job_id=a.job_id, lease_id=a.lease_id, worker_id=self.worker_id,
                               success=True, metrics_json=json.dumps(metrics), model=result.model_bytes,
                               model_filename=result.model_filename, train_seconds=elapsed)
        except JobCancelled:
            log.warning("Trabajo %s abandonado tras %.1fs", a.job_id, time.time() - t0)
            self._done(a.job_id)
            return
        except Exception as e:
            log.error("Trabajo %s falló: %s", a.job_id, e)
            msg = pb.JobResult(job_id=a.job_id, lease_id=a.lease_id, worker_id=self.worker_id,
                               success=False, error=f"{e}\n{traceback.format_exc()[-1500:]}",
                               train_seconds=time.time() - t0)
        try:
            # El resultado SÍ se reintenta con paciencia: si el coordinador se cayó,
            # se espera a que vuelva. Es seguro repetirlo porque el coordinador guarda
            # un solo resultado por job_id.
            ack = call_with_retry(lambda: self.stub.ReportResult(msg, timeout=60),
                                  what=f"ReportResult {a.job_id}", max_attempts=20, cap=15)
            log.info("Resultado de %s: %s (%s), %.1fs", a.job_id,
                     "aceptado" if ack.accepted else "descartado", ack.reason, time.time() - t0)
        except grpc.RpcError as e:
            log.error("No se pudo entregar el resultado de %s: %s", a.job_id, e.code())
        finally:
            self._done(a.job_id)

    def _done(self, job_id):
        with self.lock:
            self.running.pop(job_id, None)

    def run(self):
        hb = threading.Thread(target=self.heartbeat_loop, daemon=True, name="heartbeat")
        hb.start()   # late desde el inicio con ready=False: el coordinador lo ve pero no le da trabajos
        self.warm_up()
        self.pull_loop()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    node = Node()
    log.info("Nodo %s -> %s, slots=%s, max_jobs=%d", node.worker_id, node.addr, node.slots, node.max_jobs)
    signal.signal(signal.SIGTERM, lambda *_: node.stop.set())
    signal.signal(signal.SIGINT, lambda *_: node.stop.set())
    node.run()


if __name__ == "__main__":
    main()
