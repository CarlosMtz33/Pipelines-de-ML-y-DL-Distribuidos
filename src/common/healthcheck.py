"""Health check para el HEALTHCHECK de Docker.

  python -m common.healthcheck coordinator   -> consulta grpc.health.v1 del coordinador
  python -m common.healthcheck worker        -> el nodo está listo y su último heartbeat es reciente
"""
import os
import sys
import tempfile
import time


def coordinator() -> bool:
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    port = os.environ.get("PORT", "50051")
    with grpc.insecure_channel(f"localhost:{port}") as ch:
        resp = health_pb2_grpc.HealthStub(ch).Check(health_pb2.HealthCheckRequest(), timeout=2)
        return resp.status == health_pb2.HealthCheckResponse.SERVING


def worker() -> bool:
    path = os.environ.get("READY_FILE", os.path.join(tempfile.gettempdir(), "worker_ready"))
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < 15


if __name__ == "__main__":
    try:
        ok = {"coordinator": coordinator, "worker": worker}[sys.argv[1]]()
    except Exception as e:
        print(e)
        ok = False
    sys.exit(0 if ok else 1)
