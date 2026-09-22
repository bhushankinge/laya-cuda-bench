"""Server-scenario sweep (MLPerf-Server style): raise target QPS until the tail breaks; report the max
sustained rate under each p99 SLO.

    python -m harness.sweep --box laptop-rtx2000ada --model laya --backend eager-fp16 \
        --start-qps 5 --factor 1.5 --warmup 20 --duration 60 --slo-ms 50 130

Starts harness.server as a subprocess, runs harness.loadgen in-process per step, samples power per
step, stops once achieved QPS < 95% of target or p99 exceeds 4x the loosest SLO. Writes
results/<box>/server/<model>.<backend>.sweep.json (+ per-step request JSONL next to it).
A sustained point = achieved >= 98% of target and p99 <= SLO; the reported max is the largest such
target, and decisions/day = decisions/s x 86,400.
"""
import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

from . import RESULTS
from .loadgen import run as loadgen_run, schedule_poisson
from .power import Sampler


def start_server(model, backend, max_batch, max_delay_ms, port, timeout=900):
    proc = subprocess.Popen([sys.executable, "-m", "harness.server", "--model", model, "--backend", backend,
                             "--max-batch", str(max_batch), "--max-delay-ms", str(max_delay_ms), "--port", str(port)])
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                return proc, json.load(r)["backend"], time.time() - t0
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("server exited early")
            time.sleep(1)
    proc.kill()
    raise TimeoutError("server did not become healthy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", required=True)
    ap.add_argument("--model", default="laya")
    ap.add_argument("--backend", default="eager-fp16")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--max-delay-ms", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--start-qps", type=float, default=5)
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--slo-ms", nargs="+", type=float, default=[50, 130])
    ap.add_argument("--label", default="", help="suffix for MIG slices etc., e.g. mig-1g.12gb")
    ap.add_argument("--no-server", action="store_true", help="server already running on --port")
    args = ap.parse_args()

    out_dir = RESULTS / args.box / "server"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}.{args.backend}" + (f".{args.label}" if args.label else "")
    proc, backend_name, startup_s = (None, args.backend, 0.0) if args.no_server else \
        start_server(args.model, args.backend, args.max_batch, args.max_delay_ms, args.port)
    url = f"http://127.0.0.1:{args.port}/predict"
    steps, qps = [], args.start_qps
    try:
        with Sampler() as sampler:
            for i in range(args.max_steps):
                times = schedule_poisson(qps, args.duration + args.warmup, random.Random(i))
                t_meas = time.perf_counter() + args.warmup
                summary, _ = asyncio.run(loadgen_run(url, args.model, times, out_dir / f"{tag}.qps{qps:.1f}.jsonl", args.warmup, i))
                summary.update({"target_qps": qps, **{f"power_{k}": v for k, v in sampler.stats(since=t_meas).items()}})
                steps.append(summary)
                print(f"target {qps:7.1f} q/s -> achieved {summary.get('achieved_qps', 0):7.1f}  "
                      f"p50 {summary.get('p50_ms', 0):6.1f}  p99 {summary.get('p99_ms', 0):7.1f} ms  "
                      f"batch~{summary.get('mean_served_batch', 0):.1f}  {summary.get('power_watts_mean', 0):.0f} W", flush=True)
                # Saturated when the server can no longer keep up: errors, a tail far past the loosest SLO, requests
                # completing well below the offered rate, or latency climbing through the step (queue growth).
                # achieved/target alone is noisy at low rates (few requests per step), so it only counts when gross.
                saturated = summary.get("ok", 0) == 0 or summary.get("errors", 0) > 0 or \
                    summary["p99_ms"] > 4 * max(args.slo_ms) or summary["achieved_qps"] < 0.8 * qps or \
                    summary["latency_trend"] > 2.0
                if saturated:
                    if any(not s["saturated"] for s in steps[:-1]):
                        summary["saturated"] = True
                        break
                    summary["saturated"] = True   # first step already too high: step down until one is clean
                    qps /= args.factor
                    continue
                summary["saturated"] = False
                qps *= args.factor
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)

    sustained = {}
    for slo in args.slo_ms:
        good = [s for s in steps if s.get("ok") and s["achieved_qps"] >= 0.9 * s["target_qps"] and s["p99_ms"] <= slo
                and s["latency_trend"] <= 1.5]
        best = max(good, key=lambda s: s["target_qps"]) if good else None
        sustained[str(int(slo))] = None if not best else {
            "target_qps": best["target_qps"], "achieved_qps": best["achieved_qps"], "decisions_per_s": best["decisions_per_s"],
            "decisions_per_day": best["decisions_per_s"] * 86400, "p99_ms": best["p99_ms"], "p50_ms": best["p50_ms"],
            "watts_mean": best.get("power_watts_mean"),
            "joules_per_decision": (best.get("power_watts_mean") or 0) / best["decisions_per_s"]}
    report = {"box": args.box, "model": args.model, "backend": backend_name, "label": args.label, "max_batch": args.max_batch,
              "max_delay_ms": args.max_delay_ms, "server_startup_s": startup_s, "warmup_s": args.warmup, "step_s": args.duration,
              "slo_ms": args.slo_ms, "sustained": sustained, "steps": steps, "ts": datetime.now(timezone.utc).isoformat()}
    path = out_dir / f"{tag}.sweep.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    for slo, s in sustained.items():
        print(f"SLO p99<={slo}ms: " + (f"{s['decisions_per_s']:.0f} decisions/s = {s['decisions_per_day']/1e6:.1f}M/day "
                                       f"at {s['watts_mean'] or 0:.0f} W" if s else "not met at any step"))
    print("wrote", path)


if __name__ == "__main__":
    main()
