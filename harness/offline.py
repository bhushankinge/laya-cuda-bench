"""Offline ceiling: largest batch that fits, 50 timed iterations -> decisions/s. One backend per GPU.

    python -m harness.offline --box laptop-rtx2000ada --model laya --backend eager-fp16 --length medium
Writes results/<box>/offline/<model>.<backend>.<length>.json
"""
import argparse
import json
import time
from datetime import datetime, timezone

import torch

from . import RESULTS
from .parity import make_backend
from .power import Sampler
from .sequences import load_agent
from .shootout import run_cell
from .workload import load_workload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--model", default="laya")
    ap.add_argument("--backend", default="eager-fp16")
    ap.add_argument("--length", default="medium")
    ap.add_argument("--start", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    wl = load_workload(args.model)
    agent = load_agent(args.model)
    fn = make_backend(args.backend, agent, args.model)
    batch = args.start
    with Sampler() as sampler:
        while batch >= 1:
            t0 = time.perf_counter()
            rows, extra, err = run_cell(agent, fn, wl["buckets"][args.length], wl["questions"], batch, args.warmup, args.iters, sampler)
            if not err:
                break
            print(f"batch {batch}: {err[:60]} -> halving", flush=True)
            batch //= 2
        p = sampler.stats(since=t0)
    ms = sorted(r["ms"] for r in rows)
    dec_s = batch / (sum(ms) / len(ms)) * 1e3
    out = {"box": args.box, "gpu": torch.cuda.get_device_name(0), "model": args.model, "backend": fn.name,
           "length": args.length, "batch": batch, "iters": len(rows), "ms_mean": sum(ms) / len(ms),
           "ms_p50": ms[len(ms) // 2], "decisions_per_s": dec_s, "joules_per_decision": (p.get("watts_mean") or 0) / dec_s,
           **extra, **p, "ts": datetime.now(timezone.utc).isoformat(), "samples_ms": ms}
    path = RESULTS / args.box / "offline" / f"{args.model}.{fn.name}.{args.length}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{args.model} {fn.name} {args.length}: batch={batch} {dec_s:.0f} decisions/s {p.get('watts_mean', 0):.0f}W "
          f"{out['joules_per_decision']*1e3:.2f} mJ/decision -> {path}")


if __name__ == "__main__":
    main()
