"""Backend shootout: static latency/throughput cells, 20 warmup + 50 timed, 3 fresh-process repeats.

    python -m harness.shootout --box laptop-rtx2000ada --models laya laya-multilingual \
        --backends eager-fp16 compile-fp16 ort-cuda-fp16 ort-trt-fp16 eager-fp32 \
        --batches 1 16 64 256 --lengths short long --repeats 3

One subprocess per (model, backend, repeat) writes results/<box>/shootout/<model>/<backend>.r<k>.jsonl
(one line per timed iteration, a final {"done": true} line). Existing complete files are skipped.
Timing per iteration = prompt build + tokenize + collate + H2D + forward + D2H + calibration + formatting.
OOM is recorded as a cell with status "oom" and the run continues.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import cycle

import torch

from . import CHECKPOINTS, RESULTS
from .parity import make_backend
from .power import Sampler
from .predict import calibrate, format_answers
from .sequences import build_rows, collate, load_agent, to_device
from .workload import load_workload

DEFAULT_BACKENDS = ["eager-fp16", "compile-fp16", "ort-cuda-fp16", "ort-trt-fp16", "eager-fp32"]


def make_requests(states, questions, n):
    """n rows: states round-robin, each row one question, question types rotate route/severity/needs_human."""
    qids = list(questions)
    reqs = []
    for i, st in zip(range(n), cycle(states)):
        qid = qids[i % len(qids)]
        reqs.append((st, {qid: questions[qid]}))
    return reqs


def run_cell(agent, fn, states, questions, batch, warmup, iters, sampler=None, pad_to=None):
    """Returns (rows, error). rows = per-iteration dicts; error = None or 'oom'."""
    reqs = make_requests(states, questions, batch)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    try:
        first_call_s = None
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            items, meta = build_rows(agent, reqs)
            b = to_device(collate(agent, items, pad_to), agent.device)
            logits, act = fn(b)  # blocks on D2H copy -> includes synchronization
            probs = calibrate(agent, logits, meta)
            format_answers(probs, act, meta, len(reqs))
            ms = (time.perf_counter() - t0) * 1e3
            if i == 0:
                first_call_s = ms / 1e3
            if i >= warmup:
                rows.append({"iter": i - warmup, "ms": ms, "n_decisions": len(meta),
                             "seq_len": int(b["input_ids"].shape[1]), "tokens": int(b["attention_mask"].sum())})
        peak = torch.cuda.max_memory_allocated()
        return rows, {"first_call_s": first_call_s, "peak_vram_mb": peak / 2**20}, None
    except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:  # ORT raises its own exception types
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg or "memory" in msg or "onnxruntime" in type(e).__module__:
            torch.cuda.empty_cache()
            return [], {}, f"oom: {str(e)[:200]}"
        raise


def child(args):
    box, model, backend, rep = args.box, args.models[0], args.backends[0], args.child
    out = RESULTS / box / "shootout" / model / f"{backend}.r{rep}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    wl = load_workload(model)
    agent = load_agent(model)
    fn = make_backend(backend, agent, model)
    gpu = torch.cuda.get_device_name(0)
    with out.open("w") as f, Sampler() as sampler:
        for length in args.lengths:
            states = wl["buckets"][length]
            for batch in args.batches:
                t_start = time.perf_counter()
                rows, extra, err = run_cell(agent, fn, states, wl["questions"], batch, args.warmup, args.iters, sampler)
                p = sampler.stats(since=t_start)
                base = {"box": box, "gpu": gpu, "model": model, "backend": fn.name, "batch": batch, "length": length,
                        "repeat": rep, "ts": datetime.now(timezone.utc).isoformat()}
                if err:
                    f.write(json.dumps({**base, "status": "oom", "error": err}) + "\n")
                    print(f"{model} {fn.name} b={batch} {length}: OOM", flush=True)
                    continue
                for r in rows:
                    f.write(json.dumps({**base, **r, **extra, "watts_mean": p.get("watts_mean"),
                                        "sm_mhz_mean": p.get("sm_mhz_mean")}) + "\n")
                ms = sorted(r["ms"] for r in rows)
                print(f"{model} {fn.name} b={batch} {length}: p50={ms[len(ms)//2]:.2f}ms "
                      f"p99={ms[int(len(ms)*0.99)-1]:.2f}ms {batch/ms[len(ms)//2]*1e3:.0f} dec/s "
                      f"{p.get('watts_mean', 0):.0f}W peak={extra['peak_vram_mb']:.0f}MB", flush=True)
                f.flush()
        f.write(json.dumps({"done": True, "compile_s": getattr(fn, "compile_s", None),
                            "session_build_s": getattr(fn, "session_build_s", None)}) + "\n")


def complete(path):
    try:
        return path.exists() and json.loads(path.read_text().strip().splitlines()[-1]).get("done")
    except (json.JSONDecodeError, IndexError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", required=True)
    ap.add_argument("--models", nargs="+", default=["laya", "laya-multilingual"])
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256])
    ap.add_argument("--lengths", nargs="+", default=["short", "long"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--child", type=int, help="internal: repeat index; runs one (model, backend) in-process")
    args = ap.parse_args()
    if args.child is not None:
        return child(args)
    for model in args.models:
        for backend in args.backends:
            for rep in range(args.repeats):
                out = RESULTS / args.box / "shootout" / model / f"{backend}.r{rep}.jsonl"
                if complete(out):
                    print("skip (done)", out.relative_to(RESULTS))
                    continue
                cmd = [sys.executable, "-m", "harness.shootout", "--box", args.box, "--models", model, "--backends", backend,
                       "--batches", *map(str, args.batches), "--lengths", *args.lengths, "--warmup", str(args.warmup),
                       "--iters", str(args.iters), "--child", str(rep)]
                print("==>", model, backend, f"repeat {rep}", flush=True)
                rc = subprocess.call(cmd)
                if rc != 0:
                    print(f"child failed rc={rc}: {model} {backend} r{rep} (continuing)", flush=True)


if __name__ == "__main__":
    main()
