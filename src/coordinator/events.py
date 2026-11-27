"""Bus de eventos para el dashboard (flujo gRPC WatchEvents).

Cada suscriptor tiene una cola acotada. Si el dashboard es lento o se cuelga, su
cola se llena y se DESCARTAN sus eventos: el coordinador nunca se bloquea por él.
Además hay un máximo de suscriptores. Esto es un Bulkhead: el dashboard no puede
consumir los recursos del coordinador que necesitan los nodos.
"""
from __future__ import annotations

import json
import queue
import threading
import time


class EventBus:
    def __init__(self, max_subscribers: int = 8, queue_size: int = 2000):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.max_subscribers = max_subscribers
        self.queue_size = queue_size

    def subscribe(self) -> queue.Queue | None:
        with self._lock:
            if len(self._subs) >= self.max_subscribers:
                return None
            q = queue.Queue(maxsize=self.queue_size)
            self._subs.append(q)
            return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, job_id: str = "", worker_id: str = "", **payload) -> None:
        ev = {"ts": time.time(), "kind": kind, "job_id": job_id, "worker_id": worker_id,
              "payload_json": json.dumps(payload, default=str)}
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass   # suscriptor lento: pierde eventos, el sistema sigue
