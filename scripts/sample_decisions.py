"""Side-by-side sample of what the models actually answer: Laya (upstream Agent) vs Jev, same states, same 3 questions.

    .venv/bin/python scripts/sample_decisions.py --n 5 [--device cpu|cuda] [--model laya]
Writes results/samples/<model>-vs-jev.jsonl (full answers) and prints a compact table. Not a benchmark.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import aiohttp  # noqa: E402

from harness.jev_probe import api_key, call, jev_questions  # noqa: E402
from harness.sequences import load_agent  # noqa: E402
from harness.workload import load_workload  # noqa: E402


def compact(ans):
    if not ans:
        return "-"
    r, s, h = ans.get("route", {}), ans.get("severity", {}), ans.get("needs_human", {})
    return (f"route={r.get('choice')} ({max(r.get('probabilities', {}).values() or [0]):.2f})  "
            f"severity={s.get('score', 0):.2f}  needs_human={h.get('noul', 0):.2f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="laya")
    ap.add_argument("--no-jev", action="store_true")
    args = ap.parse_args()
    wl = load_workload("laya")
    states = [wl["buckets"]["short"][0], wl["buckets"]["medium"][0], wl["buckets"]["long"][0]] + wl["natural"][:max(0, args.n - 3)]
    agent = load_agent(args.model, device=args.device)
    out = ROOT / "results" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    async with aiohttp.ClientSession() as sess:
        for st in states[: args.n]:
            laya = agent.predict(st, wl["questions"])
            jev = None
            if not args.no_jev:
                status, ms, js = await call(sess, api_key(), st, jev_questions(wl["questions"]))
                jev = js if status == 200 else {"error": status, "body": js}
            rows.append({"state": st, "laya": laya, "jev": jev})
            print("=" * 100)
            print(st[:400].replace("\n", " ") + (" ..." if len(st) > 400 else ""))
            print(f"  LAYA ({args.model}, {args.device}, {laya['usage']['input_tokens']} tok): {compact(laya['answers'])}")
            if jev:
                print(f"  JEV  ({jev.get('model')}, {jev.get('usage', {}).get('input_tokens')} tok): {compact(jev.get('answers'))}")
    path = out / f"{args.model}-vs-jev.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nfull answers ->", path)


if __name__ == "__main__":
    asyncio.run(main())
