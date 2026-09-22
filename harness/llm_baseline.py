"""LLM baseline: the same 3 decisions as guided JSON from an OpenAI-compatible vLLM endpoint.

    python -m harness.llm_baseline --box rtxpro6000-ws --label qwen3.5-35b-a3b-fp8 \
        --base-url http://<vllm-host>:8000/v1 --llm-model <served name> --concurrency 1 16 64 --requests 60 120 240

Latency/throughput/tokens only, no accuracy claims. Token from --api-key-env (default none).
Writes results/<box>/baselines/<label>.json and per-level JSONL.
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp

from . import RESULTS
from .loadgen import summarize
from .workload import load_workload

SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["hardware", "software", "services"]},
        "severity": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    },
    "required": ["route", "severity", "needs_human"],
    "additionalProperties": False,
}


def prompt(state, questions):
    q = questions
    return ("Answer three questions about the government solicitation below. Reply with JSON only.\n"
            f"route: {q['route']['instructions']} Options: " + "; ".join(f"{k} = {v}" for k, v in q["route"]["criteria"].items()) + "\n"
            f"severity: {q['severity']['instructions']} Levels: " + "; ".join(f"{i} = {c}" for i, c in enumerate(q["severity"]["criteria"])) + "\n"
            f"needs_human: {q['needs_human']['instructions']} (true/false)\n\nSOLICITATION:\n{state}")


async def call(sess, base_url, model, key, state, questions, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt(state, questions)}], "max_tokens": 64,
            "temperature": 0, "response_format": {"type": "json_schema", "json_schema": {"name": "decisions", "schema": SCHEMA}}, **extra}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    t0 = time.perf_counter()
    try:
        async with sess.post(f"{base_url}/chat/completions", json=body, headers=headers) as r:
            js = await r.json(content_type=None)
            return r.status, (time.perf_counter() - t0) * 1e3, js
    except Exception as e:
        return 599, (time.perf_counter() - t0) * 1e3, {"error": str(e)}


async def level(args, states, questions, concurrency, n, extra):
    rows, sem = [], asyncio.Semaphore(concurrency)
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), connector=aiohttp.TCPConnector(limit=0, ssl=False)) as sess:
        start = time.perf_counter()

        async def one(i):
            async with sem:
                t_sched = time.perf_counter() - start
                status, ms, js = await call(sess, args.base_url, args.llm_model, key, states[i % len(states)], questions, extra)
                done = time.perf_counter() - start
                usage = js.get("usage") or {} if isinstance(js, dict) else {}
                content = ""
                try:
                    content = js["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    n_dec = sum(k in parsed for k in ("route", "severity", "needs_human"))
                except Exception:
                    n_dec = 0
                rows.append({"t_sched": t_sched, "t_send": t_sched, "t_done": done, "latency_ms": ms, "send_lag_ms": 0.0,
                             "status": status if n_dec == 3 else (status if status != 200 else 422), "served_batch": 1,
                             "n_decisions": n_dec, "input_tokens": usage.get("prompt_tokens", 0),
                             "output_tokens": usage.get("completion_tokens", 0)})

        await asyncio.gather(*(one(i) for i in range(n)))  # closed loop: `concurrency` clients back to back
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--api-key-env")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 16, 64])
    ap.add_argument("--requests", nargs="+", type=int, default=[60, 120, 240])
    ap.add_argument("--no-think", action="store_true", help="Qwen3: pass chat_template_kwargs enable_thinking=false")
    args = ap.parse_args()
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}
    wl = load_workload("laya")
    out_dir = RESULTS / args.box / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"box": args.box, "label": args.label, "base_url": args.base_url, "llm_model": args.llm_model, "levels": {},
              "note": "closed-loop clients; decisions = 3 per request; guided JSON via response_format json_schema",
              "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for c, n in zip(args.concurrency, args.requests):
        rows = asyncio.run(level(args, wl["natural"], wl["questions"], c, n, extra))
        with open(out_dir / f"{args.label}.c{c}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = summarize(rows)
        ok = [r for r in rows if r["status"] == 200]
        s["mean_prompt_tokens"] = sum(r["input_tokens"] for r in ok) / max(1, len(ok))
        s["mean_completion_tokens"] = sum(r["output_tokens"] for r in ok) / max(1, len(ok))
        report["levels"][str(c)] = s
        print(f"{args.label} c={c:3d}: {s.get('ok', 0)}/{n} ok  p50 {s.get('p50_ms', 0):.0f} ms  p99 {s.get('p99_ms', 0):.0f} ms  "
              f"{s.get('decisions_per_s', 0):.1f} decisions/s  prompt~{s['mean_prompt_tokens']:.0f} tok", flush=True)
    (out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", out_dir / f"{args.label}.json")


if __name__ == "__main__":
    main()
