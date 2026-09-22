"""Frozen workload loader: public solicitation texts bucketed by token length per checkpoint tokenizer.

fixtures/workload/states.jsonl lines: {"id", "text", "tokens": {"laya": n, "laya-multilingual": n, ...}}
fixtures/workload/questions.json: the 3-question bundle (route: choice, severity: score, needs_human: noul).
Buckets (by the checkpoint's own token count): short <= 80, medium 160..360, long >= 400 (<= max_len - 40),
xlong 700..984 (only meaningful for 1,024-context checkpoints).
"""
import json
import random

from . import FIXTURES

BUCKETS = {"short": (0, 80), "medium": (160, 360), "long": (400, 472), "xlong": (700, 984)}


def load_workload(model, seed=0, min_per_bucket=32):
    d = FIXTURES / "workload"
    states = [json.loads(l) for l in (d / "states.jsonl").read_text().splitlines() if l.strip()]
    questions = json.loads((d / "questions.json").read_text())
    tok_key = "laya-multilingual" if model == "laya-multilingual" else "laya"  # typed-decisions shares laya's tokenizer
    buckets = {}
    for name, (lo, hi) in BUCKETS.items():
        sel = [s["text"] for s in states if lo <= s["tokens"][tok_key] <= hi]
        random.Random(seed).shuffle(sel)
        if len(sel) >= min_per_bucket:
            buckets[name] = sel
    natural = [s["text"] for s in states if s["tokens"][tok_key] <= (984 if tok_key == "laya-multilingual" else 472)]
    return {"buckets": buckets, "natural": natural, "questions": questions, "n_states": len(states)}


if __name__ == "__main__":
    for m in ("laya", "laya-multilingual"):
        w = load_workload(m)
        print(m, {k: len(v) for k, v in w["buckets"].items()}, "natural", len(w["natural"]))
