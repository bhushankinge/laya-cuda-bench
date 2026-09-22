"""Jev (TypeSafe System One API) probe: latency and tokens/decision for the same 3-question bundle.

    python -m harness.jev_probe --verify                 # one call, prints the raw response
    python -m harness.jev_probe --concurrency 1 8 32 --requests 100 200 300

Reads the key from .env (Jev_AI=...) or TYPESAFE_API_KEY. Stays rate-limit aware: caps at --max-rps
(default 10 req/s, well under the reported 1,200 req/min) and stops a level on the first 429.
Network round trip is part of the number: say so wherever it is reported.
Writes results/jev/probe.json and per-request results/jev/c<N>.jsonl.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp

from . import RESULTS, ROOT
from .loadgen import summarize
from .workload import load_workload

URL = "https://api.typesafe.ai/v1/systemone"


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(("Jev_AI=", "TYPESAFE_API_KEY=")):
                k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        raise SystemExit("no Jev key: set TYPESAFE_API_KEY or put Jev_AI=... in .env")
    return k


def jev_questions(q):
    """Jev's request schema is the same as Laya's (lowercase types, dict criteria for choice, list for score);
    the only normalisation is a list-form choice -> dict."""
    out = {}
    for name, d in q.items():
        body = {"type": d["type"], "instructions": d["instructions"]}
        if d["type"] == "choice":
            body["criteria"] = d["criteria"] if isinstance(d["criteria"], dict) else {c: "" for c in d["criteria"]}
        elif d["type"] == "score":
            body["criteria"] = list(d["criteria"])
        out[name] = body
    return out


async def call(sess, key, state, questions, model="jev-latest"):
    t0 = time.perf_counter()
    try:
        async with sess.post(URL, json={"state": state, "questions": questions, "model": model},
                             headers={"Authorization": f"Bearer {key}"}) as r:
            js = await r.json(content_type=None)
            return r.status, (time.perf_counter() - t0) * 1e3, js
    except Exception as e:
        return 599, (time.perf_counter() - t0) * 1e3, {"error": str(e)}


async def level(key, states, questions, concurrency, n, max_rps, out_path):
    rows, sem = [], asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as sess:
        start = time.perf_counter()

        async def one(i, t_sched):
            async with sem:
                if stop.is_set():
                    return
                status, ms, js = await call(sess, key, states[i % len(states)], questions)
                done = time.perf_counter() - start
                if status == 429:
                    stop.set()
                rows.append({"t_sched": t_sched, "t_send": done - ms / 1e3, "t_done": done, "latency_ms": ms,
                             "send_lag_ms": 0.0, "status": status, "served_batch": 1,
                             "n_decisions": len(js.get("answers", {})) if isinstance(js, dict) else 0,
                             "input_tokens": (js.get("usage") or {}).get("input_tokens", 0) if isinstance(js, dict) else 0,
                             "model": js.get("model") if isinstance(js, dict) else None})

        tasks = []
        for i in range(n):
            t = i / max_rps
            delay = t - (time.perf_counter() - start)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i, t)))
        await asyncio.gather(*tasks)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    s = summarize(rows)
    s["rate_limited"] = stop.is_set()
    s["status_counts"] = {str(k): sum(1 for r in rows if r["status"] == k) for k in sorted({r["status"] for r in rows})}
    ok = [r for r in rows if r["status"] == 200]
    s["mean_input_tokens_per_request"] = sum(r["input_tokens"] for r in ok) / max(1, len(ok))
    s["mean_decisions_per_request"] = sum(r["n_decisions"] for r in ok) / max(1, len(ok))
    s["model"] = next((r["model"] for r in ok), None)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--requests", nargs="+", type=int, default=[100, 200, 300])
    ap.add_argument("--max-rps", type=float, default=10)
    args = ap.parse_args()
    key = api_key()
    wl = load_workload("laya")
    q = jev_questions(wl["questions"])
    out_dir = RESULTS / "jev"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.verify:
        async def v():
            async with aiohttp.ClientSession() as sess:
                return await call(sess, key, wl["natural"][0], q)
        status, ms, js = asyncio.run(v())
        print(status, f"{ms:.0f} ms", json.dumps(js, indent=1)[:2000])
        return
    report = {"url": URL, "max_rps_cap": args.max_rps, "levels": {}, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for c, n in zip(args.concurrency, args.requests):
        s = asyncio.run(level(key, wl["natural"], q, c, n, args.max_rps, out_dir / f"c{c}.jsonl"))
        report["levels"][str(c)] = s
        print(f"concurrency {c:3d}: {s.get('ok', 0)}/{n} ok  p50 {s.get('p50_ms', 0):.0f} ms  p95 {s.get('p95_ms', 0):.0f} ms  "
              f"{s.get('achieved_qps', 0):.1f} req/s  tokens/req {s['mean_input_tokens_per_request']:.0f}  "
              f"{'RATE LIMITED' if s['rate_limited'] else ''}", flush=True)
        if s["rate_limited"]:
            break
    (out_dir / "probe.json").write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", out_dir / "probe.json")


if __name__ == "__main__":
    main()
