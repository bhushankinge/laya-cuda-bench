"""Load results/ into pandas frames and compute the cost model. Used by notebooks/report.ipynb.

    from harness.report import shootout, sweeps, offline, baselines, jev, cost_per_million
"""
import json
from pathlib import Path

import pandas as pd

from . import RESULTS

# ---- cost model assumptions (stated in the post; override in the notebook) ----
USD_PER_KWH = 0.12
CARD_USD = {  # street price assumptions, September 2026; H100 NVL is per card
    "NVIDIA RTX 2000 Ada Generation Laptop GPU": 650, "NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU": 2500,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 8500, "NVIDIA H100 NVL": 30000,
}
LIFETIME_YEARS = 3
JEV_USD_PER_M_INPUT_TOKENS = 0.042


def _jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def shootout(box=None):
    """Per-iteration rows -> per-cell medians. median_ms is the median over repeats of per-repeat medians."""
    rows = []
    for p in RESULTS.glob(f"{box or '*'}/shootout/*/*.jsonl"):
        rows += [r for r in _jsonl(p) if "ms" in r or r.get("status") == "oom"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ok = df[df.get("status").isna()] if "status" in df else df
    per_rep = ok.groupby(["box", "gpu", "model", "backend", "batch", "length", "repeat"]).agg(
        p50_ms=("ms", "median"), p99_ms=("ms", lambda s: s.quantile(0.99)), watts=("watts_mean", "mean"),
        peak_vram_mb=("peak_vram_mb", "max"), sm_mhz=("sm_mhz_mean", "mean"), n=("ms", "size")).reset_index()
    cells = per_rep.groupby(["box", "gpu", "model", "backend", "batch", "length"]).agg(
        median_ms=("p50_ms", "median"), min_ms=("p50_ms", "min"), max_ms=("p50_ms", "max"), p99_ms=("p99_ms", "median"),
        watts=("watts", "mean"), peak_vram_mb=("peak_vram_mb", "max"), sm_mhz=("sm_mhz", "mean"), repeats=("repeat", "nunique")).reset_index()
    cells["decisions_per_s"] = cells["batch"] / cells["median_ms"] * 1e3
    cells["mj_per_decision"] = cells["watts"] / cells["decisions_per_s"] * 1e3
    return cells


def sweeps(box=None):
    out = []
    for p in RESULTS.glob(f"{box or '*'}/server/*.sweep.json"):
        d = json.loads(p.read_text())
        for slo, s in d["sustained"].items():
            out.append({"box": d["box"], "model": d["model"], "backend": d["backend"], "label": d["label"], "slo_ms": int(slo),
                        **(s or {"decisions_per_s": None})})
    return pd.DataFrame(out)


def sweep_steps(box=None):
    out = []
    for p in RESULTS.glob(f"{box or '*'}/server/*.sweep.json"):
        d = json.loads(p.read_text())
        for s in d["steps"]:
            out.append({"box": d["box"], "model": d["model"], "backend": d["backend"], "label": d["label"], **s})
    return pd.DataFrame(out)


def offline(box=None):
    return pd.DataFrame([json.loads(p.read_text()) for p in RESULTS.glob(f"{box or '*'}/offline/*.json")]).drop(columns=["samples_ms"], errors="ignore")


def baselines(box=None):
    out = []
    for p in RESULTS.glob(f"{box or '*'}/baselines/*.json"):
        d = json.loads(p.read_text())
        for c, s in d["levels"].items():
            out.append({"box": d["box"], "label": d["label"], "concurrency": int(c), **s})
    return pd.DataFrame(out)


def jev():
    p = RESULTS / "jev" / "probe.json"
    if not p.exists():
        return pd.DataFrame()
    d = json.loads(p.read_text())
    return pd.DataFrame([{"concurrency": int(c), **s} for c, s in d["levels"].items()])


def cost_per_million(decisions_per_s, watts, gpu_name, utilisation=1.0, usd_per_kwh=USD_PER_KWH, years=LIFETIME_YEARS):
    """$/M decisions = energy + amortised card. utilisation scales the lifetime decision count (1.0 = 24/7 at the rate)."""
    energy = watts / decisions_per_s / 3.6e6 * usd_per_kwh * 1e6
    lifetime_decisions = decisions_per_s * utilisation * 86400 * 365 * years
    amort = CARD_USD.get(gpu_name, float("nan")) / lifetime_decisions * 1e6
    return {"energy_usd_per_m": energy, "amortised_usd_per_m": amort, "total_usd_per_m": energy + amort}


def jev_cost_per_million(mean_input_tokens_per_request, decisions_per_request=3):
    tokens_per_decision = mean_input_tokens_per_request / decisions_per_request
    return tokens_per_decision * JEV_USD_PER_M_INPUT_TOKENS


if __name__ == "__main__":
    print(shootout().head(20).to_string() if not shootout().empty else "no shootout results yet")
    print(sweeps().to_string() if not sweeps().empty else "no sweeps yet")
    print(cost_per_million(1000, 300, "NVIDIA H100 NVL"))
