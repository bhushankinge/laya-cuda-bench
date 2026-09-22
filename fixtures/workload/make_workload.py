"""Build the frozen public workload from SAM.gov's Contract Opportunities bulk extract (no API key).

    .venv/bin/python fixtures/workload/make_workload.py [--mb 80] [--n 1000] [--seed 0]

Source: https://falextracts.s3.amazonaws.com/Contract Opportunities/datagov/ContractOpportunitiesFullCSV.csv
(the data.gov bulk file SAM.gov publishes daily). Only the first --mb megabytes are fetched with a Range
request (newest notices first). Each state = "<Title>\\n\\n<Description>", public text only. Token counts
are recorded per checkpoint tokenizer; sampling is stratified so every length bucket has >= 15% of --n.
Writes states.jsonl and source.json (ETag, Last-Modified, bytes fetched) for provenance.
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from harness.workload import BUCKETS  # noqa: E402

URL = "https://falextracts.s3.amazonaws.com/Contract%20Opportunities/datagov/ContractOpportunitiesFullCSV.csv"
TOKENIZERS = {"laya": ROOT / "models/laya/tokenizer", "laya-multilingual": ROOT / "models/laya-multilingual/tokenizer"}


def fetch(mb):
    req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{mb * 2**20 - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
        meta = {"url": URL, "etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"), "bytes": len(data)}
    return data.decode("utf-8", errors="replace"), meta


def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text)  # contact details are not needed for the decisions
    text = re.sub(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b", "[phone]", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=80)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    toks = {k: AutoTokenizer.from_pretrained(str(v)) for k, v in TOKENIZERS.items()}

    text, meta = fetch(args.mb)
    csv.field_size_limit(10**9)
    rows = list(csv.reader(io.StringIO(text)))
    hdr, body = rows[0], [r for r in rows[1:-1] if len(r) == len(rows[0])]  # drop the truncated last row
    print(f"fetched {meta['bytes']/2**20:.0f} MB, {len(body)} complete rows")
    cands, seen = [], set()
    for r in body:
        d = dict(zip(hdr, r))
        desc = clean(d.get("Description", ""))
        if len(desc) < 120 or desc in seen or d.get("CountryCode", "USA") not in ("USA", ""):
            continue
        seen.add(desc)
        state = f"{clean(d.get('Title', ''))}\n\n{desc}"
        cands.append({"notice_id": d["NoticeId"], "posted": d.get("PostedDate"), "type": d.get("Type"), "naics": d.get("NaicsCode"),
                      "text": state, "tokens": {k: len(t(state, add_special_tokens=False)["input_ids"]) for k, t in toks.items()}})
    print(f"{len(cands)} usable notices")

    rng = random.Random(args.seed)
    rng.shuffle(cands)
    # stratify on the English tokenizer's buckets; the rest fills from the natural distribution
    per_bucket = max(1, int(0.15 * args.n))
    chosen, used = [], set()
    for name, (lo, hi) in BUCKETS.items():
        pool = [c for c in cands if lo <= c["tokens"]["laya"] <= hi] if name != "xlong" else \
               [c for c in cands if lo <= c["tokens"]["laya-multilingual"] <= hi]
        take = pool[:per_bucket]
        chosen += take
        used |= {c["notice_id"] for c in take}
        print(f"  bucket {name:6s}: {len(pool):5d} available, took {len(take)}")
    for c in cands:
        if len(chosen) >= args.n:
            break
        if c["notice_id"] not in used and c["tokens"]["laya"] <= 2000:
            chosen.append(c)
            used.add(c["notice_id"])
    chosen = chosen[: args.n]
    with open(HERE / "states.jsonl", "w") as f:
        for i, c in enumerate(chosen):
            f.write(json.dumps({"id": i, **c}, ensure_ascii=False) + "\n")
    meta.update({"n": len(chosen), "seed": args.seed, "candidates": len(cands)})
    (HERE / "source.json").write_text(json.dumps(meta, indent=1) + "\n")
    cnt = Counter(next((b for b, (lo, hi) in BUCKETS.items() if lo <= c["tokens"]["laya"] <= hi), "other") for c in chosen)
    print(f"wrote {len(chosen)} states; laya-token buckets: {dict(cnt)}; source {meta}")


if __name__ == "__main__":
    main()
