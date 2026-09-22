"""Open-loop load generator: Poisson arrivals at a target QPS, or a replay curve; records every request.

    python -m harness.loadgen --url http://127.0.0.1:8080/predict --model laya --qps 50 --duration 60 --out x.jsonl
    python -m harness.loadgen ... --curve fixtures/workload/diurnal.csv --total-decisions 10000000 --compress 24

Each request = one state (drawn from the natural length distribution) + the 3-question bundle.
Open loop: send times are scheduled from the arrival process regardless of responses, so queueing shows
up as latency (closed-loop clients hide saturation). Latency = t_done - t_scheduled (includes any
client-side lag when the loop falls behind; that lag is also recorded as send_lag_ms).
"""
import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import aiohttp

from .workload import load_workload


def summarize(rows):
    ok = [r for r in rows if r["status"] == 200]
    lat = sorted(r["latency_ms"] for r in ok)
    if not lat:
        return {"requests": len(rows), "ok": 0}
    span = max(r["t_done"] for r in ok) - min(r["t_sched"] for r in ok)
    pct = lambda p: lat[min(len(lat) - 1, int(math.ceil(p * len(lat))) - 1)]
    by_time = sorted(ok, key=lambda r: r["t_sched"])
    third = max(1, len(by_time) // 3)
    med = lambda rs: sorted(r["latency_ms"] for r in rs)[len(rs) // 2]
    return {"requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok), "achieved_qps": len(ok) / span,
            "decisions_per_s": sum(r["n_decisions"] for r in ok) / span, "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "p99_ms": pct(0.99), "max_ms": lat[-1], "mean_send_lag_ms": sum(r["send_lag_ms"] for r in ok) / len(ok),
            "mean_served_batch": sum(r["served_batch"] for r in ok) / len(ok),
            # open-loop saturation signal: the queue grows within the step -> the last third is slower than the first
            "latency_trend": med(by_time[-third:]) / max(1e-9, med(by_time[:third]))}


def schedule_poisson(qps, duration, rng):
    t, out = 0.0, []
    while t < duration:
        t += rng.expovariate(qps)
        out.append(t)
    return out


def schedule_curve(curve_csv, total_decisions, compress, rng, decisions_per_request=3):
    """curve_csv: hour,weight (24 rows). Scales to total_decisions/day, compresses 24 h into 24/compress h."""
    weights = [float(l.split(",")[1]) for l in Path(curve_csv).read_text().splitlines()[1:] if l.strip()]
    total_req = total_decisions / decisions_per_request
    hour_s = 3600.0 / compress
    out, t0 = [], 0.0
    for w in weights:
        n = total_req * w / sum(weights)
        qps = n / hour_s
        t = t0
        while t < t0 + hour_s:
            t += rng.expovariate(qps)
            out.append(t)
        t0 += hour_s
    return out


async def run(url, model, times, out_path, warmup_s, seed):
    wl = load_workload(model)
    rng = random.Random(seed)
    states = wl["natural"]
    rows = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=30)) as sess:
        start = time.perf_counter()

        async def one(t_sched):
            body = {"state": rng.choice(states), "questions": wl["questions"]}
            now = time.perf_counter() - start
            try:
                async with sess.post(url, json=body) as resp:
                    js = await resp.json() if resp.status == 200 else {}
                    status = resp.status
            except Exception:
                js, status = {}, 599
            done = time.perf_counter() - start
            rows.append({"t_sched": t_sched, "t_send": now, "t_done": done, "latency_ms": (done - t_sched) * 1e3,
                         "send_lag_ms": (now - t_sched) * 1e3, "status": status,
                         "served_batch": js.get("served_batch", 0), "n_decisions": len(js.get("answers", {})),
                         "input_tokens": js.get("usage", {}).get("input_tokens", 0)})

        tasks = []
        for t in times:
            delay = t - (time.perf_counter() - start)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(t)))
        await asyncio.gather(*tasks)
    measured = [r for r in rows if r["t_sched"] >= warmup_s]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return summarize(measured), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/predict")
    ap.add_argument("--model", default="laya")
    ap.add_argument("--qps", type=float)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=0, help="seconds at the start excluded from the summary")
    ap.add_argument("--curve"); ap.add_argument("--total-decisions", type=float); ap.add_argument("--compress", type=float, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    times = schedule_curve(args.curve, args.total_decisions, args.compress, rng) if args.curve else \
        schedule_poisson(args.qps, args.duration + args.warmup, rng)
    summary, _ = asyncio.run(run(args.url, args.model, times, args.out, args.warmup, args.seed))
    print(json.dumps(summary, indent=1))


def _selftest():
    """Arrival process hits the target rate; the curve schedule totals the requested decisions."""
    rng = random.Random(1)
    n = len(schedule_poisson(100, 60, rng))
    assert 5600 <= n <= 6400, n
    curve = schedule_curve(str(Path(__file__).resolve().parents[1] / "fixtures/workload/diurnal.csv"), 300_000, 24, rng)
    assert abs(len(curve) - 100_000) < 3_000, len(curve)
    assert curve[-1] <= 3600 * 1.01, curve[-1]
    rows = [{"status": 200, "latency_ms": i, "t_sched": i / 100, "t_done": i / 100 + i / 1e3, "n_decisions": 3,
             "send_lag_ms": 0, "served_batch": 4} for i in range(1, 101)]
    s = summarize(rows)
    assert s["p50_ms"] == 50 and s["p99_ms"] == 99 and s["ok"] == 100, s
    print("loadgen selftest ok:", n, "poisson arrivals/60s @100qps;", len(curve), "curve arrivals")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
