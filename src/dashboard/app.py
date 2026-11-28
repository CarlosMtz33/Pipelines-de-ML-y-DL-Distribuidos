"""Dashboard en tiempo real (punto extra).

Corre en su propio contenedor. Se suscribe al flujo gRPC WatchEvents del coordinador
y, cada vez que llega un evento, pide una foto del estado (GetSnapshot, a lo más 2
veces por segundo). Al navegador le manda el estado por Server-Sent Events (SSE), así
la página se actualiza sola sin recargar.

Aislamiento (Bulkhead): si este proceso se cae o se satura, el coordinador no se
entera: su cola de eventos para este suscriptor se llena y los descarta. El coordinador
además limita cuántos dashboards pueden suscribirse (MAX_WATCHERS).

Variables: COORDINATOR_ADDR=coordinator:50051  PORT=8080
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc

import trainer_pb2 as pb
import trainer_pb2_grpc as pb_grpc
from common.retry import backoff_delay

log = logging.getLogger("dashboard")
HERE = os.path.dirname(os.path.abspath(__file__))


class State:
    def __init__(self, addr):
        self.stub = pb_grpc.CoordinatorStub(grpc.insecure_channel(addr))
        self.cond = threading.Condition()
        self.version = 0
        self.snapshot = {"workers": [], "jobs": [], "server_time": 0}
        self.events: list[dict] = []
        self.extra: dict[str, dict] = {}     # job_id -> última métrica de validación reportada
        self.connected = False
        self._dirty = threading.Event()

    # -------------------------------------------------- hilos de fondo
    def watch_loop(self):
        """Suscripción al flujo de eventos; si se corta, reconecta con backoff."""
        attempt = 0
        while True:
            try:
                for ev in self.stub.WatchEvents(pb.WatchRequest()):
                    attempt = 0
                    self.connected = True
                    self._on_event(ev)
            except grpc.RpcError as e:
                self.connected = False
                attempt += 1
                log.warning("flujo de eventos cortado (%s); reintento", e.code().name)
                self._dirty.set()
                time.sleep(backoff_delay(min(attempt, 5), 0.5, 5.0))

    def snapshot_loop(self):
        """Refresca la foto cuando hay eventos (máx. 2/s) y cada 3 s aunque no haya."""
        while True:
            self._dirty.wait(timeout=3.0)
            self._dirty.clear()
            try:
                snap = self.stub.GetSnapshot(pb.SnapshotRequest(), timeout=3)
                self._set_snapshot(snap)
            except grpc.RpcError:
                self.connected = False
                self._publish()
            time.sleep(0.5)

    # -------------------------------------------------- estado
    def _on_event(self, ev):
        payload = json.loads(ev.payload_json or "{}")
        if ev.kind == "job_progress":
            self.extra[ev.job_id] = {k: v for k, v in payload.items()
                                     if k not in ("epoch", "total", "loss")}
        if ev.kind != "job_progress" and ev.kind != "job_checkpoint":
            item = {"ts": ev.ts, "kind": ev.kind, "job_id": ev.job_id,
                    "worker_id": ev.worker_id, "payload": payload}
            with self.cond:
                self.events = (self.events + [item])[-80:]
        self._dirty.set()

    def _set_snapshot(self, snap):
        workers = [{"worker_id": w.worker_id, "hostname": w.hostname, "alive": w.alive,
                    "ready": w.ready, "last_seen": w.last_seen, "running": list(w.running_job_ids),
                    "completed": w.completed, "failed": w.failed, "slots": dict(w.slots)}
                   for w in snap.workers]
        jobs = []
        for j in snap.jobs:
            m = json.loads(j.metrics_json) if j.metrics_json else {}
            jobs.append({"job_id": j.job_id, "model": j.model, "state": j.state,
                         "worker_id": j.worker_id, "attempts": j.attempts,
                         "reassignments": j.reassignments, "epoch": j.epoch,
                         "total_epochs": j.total_epochs, "loss": j.loss,
                         "checkpoint_epoch": j.checkpoint_epoch, "train_seconds": j.train_seconds,
                         "config": json.loads(j.config_json), "error": j.error[:200],
                         "metrics": {k: v for k, v in m.items() if not isinstance(v, list)},
                         "confusion": m.get("test_confusion"), "classes": m.get("classes"),
                         "live": self.extra.get(j.job_id, {})})
        with self.cond:
            self.snapshot = {"workers": workers, "jobs": jobs, "server_time": snap.server_time}
            self.connected = True
        self._publish()

    def _publish(self):
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def payload(self):
        with self.cond:
            return json.dumps({"connected": self.connected, **self.snapshot, "events": self.events})


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with state.cond:
                            if state.version == seen:
                                state.cond.wait(timeout=15)
                            changed, seen = state.version != seen, state.version
                        data = state.payload() if changed else ""
                        msg = f"data: {data}\n\n" if changed else ": keepalive\n\n"
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_error(404)
    return Handler


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state = State(os.environ.get("COORDINATOR_ADDR", "coordinator:50051"))
    threading.Thread(target=state.watch_loop, daemon=True).start()
    threading.Thread(target=state.snapshot_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(state))
    server.daemon_threads = True
    log.info("Dashboard en http://localhost:%d", port)
    server.serve_forever()


if __name__ == "__main__":
    main()
