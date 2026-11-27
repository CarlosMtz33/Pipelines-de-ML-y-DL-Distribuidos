"""Coordinador: recibe peticiones, mantiene la cola y reparte trabajos a los nodos.

Patrones que viven aquí:
  - Coordinador con cola de trabajos: store.py (SQLite) + Submit/PullJob/ReportResult.
  - Health check y heartbeat: Heartbeat() registra a los nodos; el "reaper" marca
    caído a un nodo que no late en HEARTBEAT_MISSES intervalos y reencola sus trabajos.
  - Timeout: cada asignación tiene un plazo según el modelo; si vence, se reencola.
  - Idempotencia: job_id determinista, envío repetido -> mismo trabajo, un solo
    resultado por job_id, escritura atómica de resultados.
  - Consumidores competidores: los nodos piden trabajo (PullJob) cuando tienen espacio.

Variables de entorno (con su valor por defecto):
  PORT=50051  STATE_DIR=/state  DATA_DIR=/data  HEARTBEAT_INTERVAL_MS=2000
  HEARTBEAT_MISSES=3  TIMEOUT_LINEAR=120  TIMEOUT_MLP=900  TIMEOUT_CNN=1800
  MAX_ATTEMPTS=4  SCHEDULING=lpt  MAX_WATCHERS=8  RESTART_GRACE_S=20
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import uuid
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

import trainer_pb2 as pb
import trainer_pb2_grpc as pb_grpc
from common import models as M
from common.files import atomic_write_bytes, atomic_write_json, read_data_version
from coordinator.events import EventBus
from coordinator.store import DONE, FAILED, PENDING, RUNNING, JobStore

log = logging.getLogger("coordinator")


def env_f(name, default):
    return float(os.environ.get(name, default))


class Settings:
    def __init__(self):
        self.port = int(os.environ.get("PORT", 50051))
        self.state_dir = os.environ.get("STATE_DIR", "/state")
        self.data_dir = os.environ.get("DATA_DIR", "/data")
        self.hb_interval = env_f("HEARTBEAT_INTERVAL_MS", 2000) / 1000.0
        self.hb_misses = int(os.environ.get("HEARTBEAT_MISSES", 3))
        self.timeouts = {"linear": env_f("TIMEOUT_LINEAR", 120),
                         "mlp": env_f("TIMEOUT_MLP", 900),
                         "cnn": env_f("TIMEOUT_CNN", 1800)}
        self.max_attempts = int(os.environ.get("MAX_ATTEMPTS", 4))
        self.lpt = os.environ.get("SCHEDULING", "lpt").lower() == "lpt"
        self.max_watchers = int(os.environ.get("MAX_WATCHERS", 8))
        # Tras reiniciar el coordinador, cuánto se espera a que los nodos vuelvan a latir
        # antes de reencolar los trabajos RUNNING que tenían asignados.
        self.restart_grace = env_f("RESTART_GRACE_S", 20)


def job_cost(model: str, cfg: dict) -> float:
    """Estimación gruesa del costo de un trabajo, solo para ordenar la cola (LPT)."""
    if model == "linear":
        return 1.0
    if model == "mlp":
        return 10.0 + cfg["epochs"] * sum(cfg["hidden"]) / 50.0
    return 1000.0 + cfg["epochs"] * cfg["filters"] * (cfg["img_size"] / 32) ** 2 * cfg["conv_layers"]


class WorkerState:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.hostname = ""
        self.alive = True
        self.ready = False
        self.accepted = False
        self.last_seen = 0.0
        self.running: list[str] = []
        self.slots: dict[str, int] = {}
        self.completed = 0
        self.failed = 0


class CoordinatorService(pb_grpc.CoordinatorServicer):
    def __init__(self, settings: Settings):
        self.s = settings
        os.makedirs(os.path.join(settings.state_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(settings.state_dir, "checkpoints"), exist_ok=True)
        self.store = JobStore(os.path.join(settings.state_dir, "coordinator.db"))
        self.data_version = read_data_version(settings.data_dir)
        self.events = EventBus(max_subscribers=settings.max_watchers)
        self.workers: dict[str, WorkerState] = {}
        self.wlock = threading.Lock()
        self.started_at = time.time()
        self._stop = threading.Event()
        n_running = len(self.store.running())
        log.info("Coordinador listo. Datos=%s. Trabajos RUNNING recuperados del disco: %d",
                 self.data_version, n_running)

    # =========================================================== utilidades
    def _paths(self, job_id):
        base = os.path.join(self.s.state_dir, "results", job_id)
        return base, os.path.join(self.s.state_dir, "checkpoints", f"{job_id}.ckpt")

    def _timeout_for(self, row) -> float:
        cfg = json.loads(row["config_json"])
        return float(cfg.get("timeout_s") or self.s.timeouts[row["model"]])

    @staticmethod
    def _job_info(row) -> pb.JobInfo:
        return pb.JobInfo(
            job_id=row["job_id"], model=row["model"], config_json=row["config_json"],
            seed=row["seed"], data_version=row["data_version"], state=row["state"],
            worker_id=row["worker_id"] or "", attempts=row["attempts"],
            reassignments=row["reassignments"], epoch=row["epoch"],
            total_epochs=row["total_epochs"], loss=row["loss"] or 0.0,
            metrics_json=row["metrics_json"] or "", error=row["error"] or "",
            created_at=row["created_at"], started_at=row["started_at"],
            finished_at=row["finished_at"], train_seconds=row["train_seconds"],
            checkpoint_epoch=row["checkpoint_epoch"])

    def _requeue(self, job_id, reason, expected_lease=None):
        row = self.store.get(job_id)
        new_state = self.store.requeue(job_id, reason, self.s.max_attempts, expected_lease)
        if new_state is None:
            return
        worker = row["worker_id"] if row else ""
        log.warning("Trabajo %s reencolado (%s) -> %s", job_id, reason, new_state)
        self.events.publish("job_requeued" if new_state == PENDING else "job_failed",
                            job_id, worker, reason=reason, state=new_state)

    # =========================================================== usuario
    def Submit(self, req, ctx):
        try:
            model = M.canonical_model(req.model)
            base = json.loads(req.config_json) if req.config_json else {}
            grid = json.loads(req.grid_json) if req.grid_json else None
            configs = [M.validate_config(model, c) for c in M.expand_grid(base, grid)]
        except (M.ConfigError, json.JSONDecodeError) as e:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        refs, ids = [], []
        for cfg in configs:
            job_id = M.compute_job_id(model, cfg, self.data_version, req.seed)
            if job_id in ids:
                continue
            row, existed = self.store.add_job(
                job_id=job_id, model=model, config_json=M.canonical_json(cfg), seed=req.seed,
                data_version=self.data_version, cost=job_cost(model, cfg),
                total_epochs=M.total_epochs(model, cfg))
            ids.append(job_id)
            refs.append(pb.JobRef(job_id=job_id, state=row["state"], already_existed=existed,
                                  config_json=row["config_json"]))
            if existed:
                self.events.publish("job_duplicate_submit", job_id, state=row["state"])
            else:
                self.events.publish("job_submitted", job_id, model=model, config=cfg)
        request_id = M.compute_request_id(ids)
        self.store.add_request(request_id, req.name, model, ids)
        new = sum(not r.already_existed for r in refs)
        log.info("Submit %s (%s, %s): %d trabajos, %d nuevos, %d ya existían",
                 request_id, model, req.name, len(refs), new, len(refs) - new)
        return pb.SubmitReply(request_id=request_id, jobs=refs)

    def GetJob(self, req, ctx):
        row = self.store.get(req.job_id)
        if row is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"no existe el trabajo {req.job_id}")
        return self._job_info(row)

    def ListJobs(self, req, ctx):
        return pb.JobList(jobs=[self._job_info(r) for r in self.store.list(req.request_id, req.state)])

    def GetRequest(self, req, ctx):
        meta = self.store.get_request(req.request_id)
        if meta is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"no existe la petición {req.request_id}")
        rows = self.store.list(req.request_id)
        counts = {s: sum(r["state"] == s for r in rows) for s in (PENDING, RUNNING, DONE, FAILED)}
        best_id, best_txt = "", ""
        metric, higher = M.selection_metric(meta["model"])
        done = [(json.loads(r["metrics_json"]).get(metric), r["job_id"])
                for r in rows if r["state"] == DONE and r["metrics_json"]]
        done = [d for d in done if d[0] is not None]
        if done:
            val, best_id = (max if higher else min)(done)
            best_txt = f"{metric}={val:.6g}"
        return pb.RequestStatus(request_id=req.request_id, name=meta["name"] or "",
                                pending=counts[PENDING], running=counts[RUNNING],
                                done=counts[DONE], failed=counts[FAILED],
                                jobs=[self._job_info(r) for r in rows],
                                best_job_id=best_id, best_metric=best_txt)

    def GetModel(self, req, ctx):
        row = self.store.get(req.job_id)
        if row is None or row["state"] != DONE:
            ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, "el trabajo no ha terminado")
        base, _ = self._paths(req.job_id)
        with open(os.path.join(base, row["model_filename"]), "rb") as f:
            data = f.read()
        return pb.ModelBlob(job_id=req.job_id, filename=row["model_filename"], data=data,
                            metrics_json=row["metrics_json"])

    # =========================================================== nodos
    def Heartbeat(self, req, ctx):
        now = time.time()
        with self.wlock:
            w = self.workers.get(req.worker_id)
            if w is None:
                w = self.workers[req.worker_id] = WorkerState(req.worker_id)
                log.info("Nodo nuevo: %s (%s)", req.worker_id, req.hostname)
                self.events.publish("worker_up", worker_id=req.worker_id, hostname=req.hostname)
            elif not w.alive:
                log.info("Nodo %s volvió a latir", req.worker_id)
                self.events.publish("worker_up", worker_id=req.worker_id, hostname=req.hostname,
                                    revived=True)
            w.alive, w.last_seen, w.hostname = True, now, req.hostname
            w.accepted = req.data_version == self.data_version
            w.ready = req.ready and w.accepted
            w.running = list(req.running_job_ids)
            w.slots = dict(req.slots)

        # ¿Qué trabajos reporta el nodo que ya no son suyos? -> que los cancele.
        cancel = []
        for job_id in req.running_job_ids:
            row = self.store.get(job_id)
            if row is None or row["state"] != RUNNING or row["worker_id"] != req.worker_id:
                cancel.append(job_id)
        # ¿Qué trabajos tiene asignados el nodo pero ya no está corriendo? (p. ej. se
        # reinició el contenedor). Se reencolan, con un margen por si la asignación es reciente.
        for row in self.store.running():
            if (row["worker_id"] == req.worker_id and row["job_id"] not in req.running_job_ids
                    and now - row["assigned_at"] > 2 * self.s.hb_interval):
                self._requeue(row["job_id"], "el nodo ya no lo está corriendo")

        msg = "" if w.accepted else (f"versión de datos distinta: nodo={req.data_version} "
                                     f"coordinador={self.data_version}")
        return pb.HeartbeatReply(interval_ms=int(self.s.hb_interval * 1000),
                                 cancel_job_ids=cancel, accepted=w.accepted, message=msg)

    def PullJob(self, req, ctx):
        with self.wlock:
            w = self.workers.get(req.worker_id)
            if w is None or not w.alive:
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, "nodo no registrado: mande heartbeat")
            if not w.ready:
                # Un nodo que no está listo (o con otros datos) no recibe trabajos.
                return pb.PullReply(has_job=False)
        models = [m for m in req.accept_models if m in M.MODELS]
        lease_id = uuid.uuid4().hex[:12]
        claimed = self.store.claim_next(models, req.worker_id, lease_id, self._timeout_for, self.s.lpt,
                                        avoid_grace=3 * self.s.hb_interval)
        if claimed is None:
            return pb.PullReply(has_job=False)
        row, timeout = claimed
        _, ckpt_path = self._paths(row["job_id"])
        ckpt, ckpt_epoch = b"", 0
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                ckpt = f.read()
            ckpt_epoch = row["checkpoint_epoch"]
        with self.wlock:
            w.running = list(set(w.running) | {row["job_id"]})
        log.info("Trabajo %s (%s) -> %s, intento %d, plazo %.0fs%s", row["job_id"], row["model"],
                 req.worker_id, row["attempts"], timeout,
                 f", reanuda desde época {ckpt_epoch}" if ckpt else "")
        self.events.publish("job_started", row["job_id"], req.worker_id, model=row["model"],
                            attempt=row["attempts"], resume_epoch=ckpt_epoch)
        return pb.PullReply(has_job=True, job=pb.Assignment(
            job_id=row["job_id"], lease_id=lease_id, model=row["model"],
            config_json=row["config_json"], seed=row["seed"], data_version=row["data_version"],
            timeout_s=timeout, attempt=row["attempts"], checkpoint=ckpt,
            checkpoint_epoch=ckpt_epoch))

    def ReportProgress(self, req, ctx):
        ok = self.store.progress(req.job_id, req.lease_id, req.epoch, req.total_epochs, req.loss)
        if ok:
            extra = json.loads(req.extra_json) if req.extra_json else {}
            self.events.publish("job_progress", req.job_id, req.worker_id, epoch=req.epoch,
                                total=req.total_epochs, loss=req.loss, **extra)
        return pb.Ack(ok=ok, cancel=not ok)

    def SaveCheckpoint(self, req, ctx):
        if not self.store.lease_is_current(req.job_id, req.lease_id):
            return pb.Ack(ok=False, cancel=True)
        _, ckpt_path = self._paths(req.job_id)
        atomic_write_bytes(ckpt_path, req.data)
        self.store.set_checkpoint_epoch(req.job_id, req.epoch)
        self.events.publish("job_checkpoint", req.job_id, req.worker_id, epoch=req.epoch)
        return pb.Ack(ok=True)

    def ReportResult(self, req, ctx):
        row = self.store.get(req.job_id)
        if row is None:
            return pb.ResultAck(accepted=False, reason="trabajo desconocido")
        if row["state"] == DONE:
            # Resultado único: ya hay uno guardado para este job_id. Este se descarta.
            log.info("Resultado duplicado de %s para %s: descartado", req.worker_id, req.job_id)
            self.events.publish("result_discarded", req.job_id, req.worker_id, reason="duplicado")
            return pb.ResultAck(accepted=False, reason="duplicado")

        if not req.success:
            if row["lease_id"] == req.lease_id:
                self._bump_worker(req.worker_id, failed=True)
                self._requeue(req.job_id, f"error en {req.worker_id}: {req.error[:500]}", req.lease_id)
                return pb.ResultAck(accepted=True, reason="error registrado")
            return pb.ResultAck(accepted=False, reason="lease vencido")

        # Éxito. Se acepta el PRIMER resultado válido, venga de la asignación actual o de
        # una anterior (el entrenamiento es determinista, así que ambos serían iguales).
        base, ckpt_path = self._paths(req.job_id)
        atomic_write_bytes(os.path.join(base, req.model_filename), req.model)
        metrics = json.loads(req.metrics_json)
        atomic_write_json(os.path.join(base, "metrics.json"),
                          {"job_id": req.job_id, "worker_id": req.worker_id,
                           "train_seconds": req.train_seconds, "metrics": metrics})
        if not self.store.finish(req.job_id, req.worker_id, req.metrics_json,
                                 req.model_filename, req.train_seconds):
            self.events.publish("result_discarded", req.job_id, req.worker_id, reason="duplicado")
            return pb.ResultAck(accepted=False, reason="duplicado")
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
        self._bump_worker(req.worker_id, failed=False)
        stale = row["lease_id"] != req.lease_id
        log.info("Trabajo %s terminado por %s en %.1fs%s", req.job_id, req.worker_id,
                 req.train_seconds, " (de una asignación anterior)" if stale else "")
        self.events.publish("job_done", req.job_id, req.worker_id, train_seconds=req.train_seconds,
                            metrics={k: v for k, v in metrics.items() if not isinstance(v, list)})
        return pb.ResultAck(accepted=True, reason="ok")

    def _bump_worker(self, worker_id, failed):
        with self.wlock:
            w = self.workers.get(worker_id)
            if w:
                if failed:
                    w.failed += 1
                else:
                    w.completed += 1

    # =========================================================== dashboard
    def GetSnapshot(self, req, ctx):
        with self.wlock:
            workers = [pb.WorkerInfo(worker_id=w.worker_id, hostname=w.hostname, alive=w.alive,
                                     ready=w.ready, last_seen=w.last_seen,
                                     running_job_ids=w.running, completed=w.completed,
                                     failed=w.failed, slots=w.slots)
                       for w in self.workers.values()]
        return pb.Snapshot(workers=workers, jobs=[self._job_info(r) for r in self.store.list()],
                           server_time=time.time())

    def WatchEvents(self, req, ctx):
        q = self.events.subscribe()
        if q is None:
            ctx.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "demasiados dashboards conectados")
        try:
            while ctx.is_active() and not self._stop.is_set():
                try:
                    ev = q.get(timeout=1.0)
                except Exception:
                    continue
                yield pb.Event(**ev)
        finally:
            self.events.unsubscribe(q)

    # =========================================================== vigilante (reaper)
    def reaper_loop(self):
        """Cada medio segundo: detecta nodos caídos y trabajos con plazo vencido."""
        while not self._stop.wait(0.5):
            try:
                self._reap_once()
            except Exception:
                log.exception("error en el vigilante")

    def _reap_once(self):
        now = time.time()
        limit = self.s.hb_interval * self.s.hb_misses
        dead = []
        with self.wlock:
            for w in self.workers.values():
                if w.alive and now - w.last_seen > limit:
                    w.alive, w.ready = False, False
                    dead.append(w.worker_id)
            known = {wid for wid, w in self.workers.items() if w.alive}
        for wid in dead:
            log.warning("Nodo %s CAÍDO: %d heartbeats sin llegar", wid, self.s.hb_misses)
            self.events.publish("worker_down", worker_id=wid)

        for row in self.store.running():
            if row["worker_id"] in dead:
                self._requeue(row["job_id"], f"nodo {row['worker_id']} caído")
            elif now > row["lease_deadline"]:
                self._requeue(row["job_id"], f"timeout ({self._timeout_for(row):.0f}s) en {row['worker_id']}")
            elif row["worker_id"] not in known and now - self.started_at > max(limit, self.s.restart_grace):
                # Tras reiniciar el coordinador: el nodo dueño nunca volvió a latir.
                self._requeue(row["job_id"], f"nodo {row['worker_id']} no volvió tras reinicio")

    def stop(self):
        self._stop.set()


def serve():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    service = CoordinatorService(settings)
    opts = [("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            # Sin SO_REUSEPORT: si ya hay un coordinador en el puerto, el segundo falla
            # en vez de repartirse las conexiones con el primero sin que nadie lo note.
            ("grpc.so_reuseport", 0)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64), options=opts)
    pb_grpc.add_CoordinatorServicer_to_server(service, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(f"[::]:{settings.port}")
    server.start()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    threading.Thread(target=service.reaper_loop, daemon=True, name="reaper").start()
    log.info("Escuchando en :%d", settings.port)

    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    # Espera con plazo corto en un ciclo (y no done.wait() sin plazo): en Windows una
    # espera sin plazo no deja que Ctrl+C la interrumpa.
    try:
        while not done.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    log.info("Deteniendo coordinador")
    health_servicer.enter_graceful_shutdown()
    service.stop()
    server.stop(grace=3).wait()


if __name__ == "__main__":
    serve()
