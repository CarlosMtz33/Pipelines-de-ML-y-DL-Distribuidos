"""Cliente de línea de comandos.

Ejemplos:
  python -m client.cli submit --model cnn --config '{"filters": 16, "epochs": 5}'
  python -m client.cli submit --model linear --grid '{"alpha": [0.1, 1], "variant": ["ridge", "lasso"]}'
  python -m client.cli batch grids/referencia.json --wait
  python -m client.cli request req-1234abcd
  python -m client.cli status 3f2a...        (un trabajo)
  python -m client.cli list --state RUNNING
  python -m client.cli download 3f2a... --out modelos/
  python -m client.cli bitacora req-1234abcd req-5678... --id EXP-01   (tabla para la bitácora)

Variable de entorno: COORDINATOR_ADDR (por defecto localhost:50051).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import grpc

import trainer_pb2 as pb
import trainer_pb2_grpc as pb_grpc
from common.retry import call_with_retry


def stub(addr=None):
    addr = addr or os.environ.get("COORDINATOR_ADDR", "localhost:50051")
    ch = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)])
    return pb_grpc.CoordinatorStub(ch)


def submit(s, model, config=None, grid=None, seed=42, name=""):
    req = pb.SubmitRequest(model=model, config_json=json.dumps(config or {}),
                           grid_json=json.dumps(grid) if grid else "", seed=seed, name=name)
    # Reintentar Submit es seguro: el mismo trabajo da el mismo job_id.
    return call_with_retry(lambda: s.Submit(req, timeout=30), what="Submit")


def submit_batch(s, spec: dict):
    """Envía un archivo de lote: {"name", "seed", "requests": [{"model", "config", "grid"}]}."""
    replies = []
    for r in spec["requests"]:
        rep = submit(s, r["model"], r.get("config"), r.get("grid"),
                     r.get("seed", spec.get("seed", 42)), r.get("name", spec.get("name", "")))
        replies.append((r["model"], rep))
    return replies


def wait_requests(s, request_ids, poll=1.0, quiet=False):
    last = ""
    while True:
        sts = [call_with_retry(lambda rid=rid: s.GetRequest(pb.RequestQuery(request_id=rid), timeout=10),
                               what="GetRequest", max_attempts=0) for rid in request_ids]
        tot = {k: sum(getattr(x, k) for x in sts) for k in ("pending", "running", "done", "failed")}
        line = (f"pendientes={tot['pending']} corriendo={tot['running']} "
                f"terminados={tot['done']} fallidos={tot['failed']}")
        if not quiet and line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if tot["pending"] == 0 and tot["running"] == 0:
            return sts
        time.sleep(poll)


def print_request(st):
    print(f"{st.request_id} ({st.name}): pend={st.pending} corr={st.running} "
          f"term={st.done} fall={st.failed}")
    for j in st.jobs:
        m = json.loads(j.metrics_json) if j.metrics_json else {}
        key = "val_acc" if j.model == "cnn" else "val_rmse"
        val = f"{key}={m[key]:.4f}" if key in m else ""
        print(f"  {j.job_id}  {j.model:6s} {j.state:8s} nodo={j.worker_id or '-':14s} "
              f"intentos={j.attempts} reasig={j.reassignments} {j.train_seconds:7.1f}s {val}  {j.config_json}")
    if st.best_job_id:
        print(f"  MEJOR: {st.best_job_id} ({st.best_metric})")


def bitacora_markdown(sts, exp_id):
    """Tabla en Markdown con los campos de 'Resultados' de la bitácora, lista para pegar."""
    out = [f"### {exp_id}: resultados", "",
           "| job_id | Modelo | Configuración | Semilla | Validación | Prueba | Tiempo (s) | Nodo | Reasig. |",
           "|---|---|---|---|---|---|---|---|---|"]
    for st in sts:
        for j in st.jobs:
            m = json.loads(j.metrics_json) if j.metrics_json else {}
            cfg = json.loads(j.config_json)
            cfg = {k: v for k, v in cfg.items() if k not in ("checkpoint_every", "timeout_s")}
            if j.model == "cnn":
                val = f"acc={m.get('val_acc', float('nan')):.4f}"
                test = f"acc={m.get('test_acc', float('nan')):.4f}, bal={m.get('test_balanced_acc', float('nan')):.4f}"
            else:
                val = f"RMSE={m.get('val_rmse', float('nan')):.3f}, R²={m.get('val_r2', float('nan')):.3f}"
                test = f"RMSE={m.get('test_rmse', float('nan')):.3f}, R²={m.get('test_r2', float('nan')):.3f}"
            out.append(f"| `{j.job_id}` | {j.model} | `{json.dumps(cfg, separators=(',', ':'))}` | {j.seed} "
                       f"| {val} | {test} | {j.train_seconds:.1f} | {j.worker_id} | {j.reassignments} |")
    out.append("")
    for st in sts:
        if st.best_job_id:
            out.append(f"- Mejor de `{st.request_id}`: `{st.best_job_id}` ({st.best_metric})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(prog="client")
    ap.add_argument("--addr", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="{}")
    p.add_argument("--grid", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("batch")
    p.add_argument("file")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("request")
    p.add_argument("request_id")
    p = sub.add_parser("wait")
    p.add_argument("request_ids", nargs="+")
    p = sub.add_parser("status")
    p.add_argument("job_id")
    p = sub.add_parser("list")
    p.add_argument("--request", default="")
    p.add_argument("--state", default="")
    p = sub.add_parser("bitacora")
    p.add_argument("request_ids", nargs="+")
    p.add_argument("--id", default="EXP-XX")
    p = sub.add_parser("download")
    p.add_argument("job_id")
    p.add_argument("--out", default="modelos")

    a = ap.parse_args()
    s = stub(a.addr)
    try:
        if a.cmd == "submit":
            rep = submit(s, a.model, json.loads(a.config), json.loads(a.grid) if a.grid else None,
                         a.seed, a.name)
            print(f"request_id={rep.request_id}")
            for j in rep.jobs:
                print(f"  {j.job_id} {j.state}{'  (ya existía)' if j.already_existed else ''}  {j.config_json}")
            if a.wait:
                print_request(wait_requests(s, [rep.request_id])[0])
        elif a.cmd == "batch":
            with open(a.file) as f:
                spec = json.load(f)
            reps = submit_batch(s, spec)
            for model, rep in reps:
                new = sum(not j.already_existed for j in rep.jobs)
                print(f"{model}: request_id={rep.request_id} trabajos={len(rep.jobs)} nuevos={new}")
            if a.wait:
                for st in wait_requests(s, [r.request_id for _, r in reps]):
                    print_request(st)
        elif a.cmd == "request":
            print_request(s.GetRequest(pb.RequestQuery(request_id=a.request_id), timeout=10))
        elif a.cmd == "wait":
            for st in wait_requests(s, a.request_ids):
                print_request(st)
        elif a.cmd == "status":
            j = s.GetJob(pb.JobQuery(job_id=a.job_id), timeout=10)
            print(json.dumps({f.name: v for f, v in j.ListFields()}, indent=2, ensure_ascii=False))
        elif a.cmd == "list":
            for j in s.ListJobs(pb.ListQuery(request_id=a.request, state=a.state), timeout=10).jobs:
                print(f"{j.job_id} {j.model:6s} {j.state:8s} {j.worker_id or '-':14s} "
                      f"ep={j.epoch}/{j.total_epochs} reasig={j.reassignments}")
        elif a.cmd == "bitacora":
            sts = [s.GetRequest(pb.RequestQuery(request_id=r), timeout=10) for r in a.request_ids]
            print(bitacora_markdown(sts, a.id))
        elif a.cmd == "download":
            blob = s.GetModel(pb.JobQuery(job_id=a.job_id), timeout=60)
            d = os.path.join(a.out, a.job_id)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, blob.filename), "wb") as f:
                f.write(blob.data)
            with open(os.path.join(d, "metrics.json"), "w") as f:
                f.write(blob.metrics_json)
            print(f"Guardado en {d}/")
    except grpc.RpcError as e:
        print(f"Error {e.code().name}: {e.details()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
