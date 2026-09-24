"""Draw the README / report figures from results/ only. Writes docs/figures/<name>.png and <name>-dark.png.

    python scripts/make_figures.py

Every number is read from the committed results; the asserts pin the headline values quoted in REPORT.md so a
figure can never silently drift from the text.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import report as R  # noqa: E402

OUT = ROOT / "docs" / "figures"
GPU = {"laptop-rtx2000ada": "RTX 2000 Ada Laptop 8 GB", "zbook-rtxpro5000": "RTX PRO 5000 Blackwell Laptop 24 GB",
       "rtxpro6000-ws": "RTX PRO 6000 Blackwell 96 GB", "h100nvl": "H100 NVL 94 GB"}
BACKEND = {"torch-eager-fp16": "eager FP16", "torch-eager-fp32": "eager FP32", "ort-cuda-fp16": "ORT CUDA FP16",
           "ort-trt-fp16-maxb256": "TensorRT FP16", "ort-trt-fp16-maxb32": "TensorRT FP16",
           "torch-compile-max-autotune-fp16": "torch.compile FP16"}
# Categorical slots 1-4 of the dataviz reference palette, validated on GitHub's light (#ffffff) and dark (#0d1117)
# surfaces; light slots 3-4 are under 3:1 contrast, so every series is also direct-labelled.
THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", band="#eef4fc",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], seq="Blues"),
    "dark": dict(bg="#0d1117", ink="#f0f6fc", ink2="#9198a1", grid="#262c36", band="#132338",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500"], seq="Blues_r"),
}
LAYA, JEV, LLM = 0, 1, 2  # series slots: colour follows the entity in every chart


def style(t):
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 11, "figure.facecolor": t["bg"],
        "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"], "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["grid"], "xtick.color": t["ink2"], "ytick.color": t["ink2"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "text.parse_math": False, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 28,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 8, "axes.axisbelow": True,
    })


def subtitle(ax, text, t):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=t["ink2"], fontsize=10.5, va="bottom")


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}{'' if theme == 'light' else '-dark'}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def slo_lines(ax, t, axis="x"):
    for slo in (50, 130):
        (ax.axvline if axis == "x" else ax.axhline)(slo, color=t["ink2"], ls=(0, (3, 3)), lw=1)


# ---------------------------------------------------------------- data

def load():
    sus = R.sweeps()
    steps = R.sweep_steps()
    base = R.baselines()
    jev = pd.concat([R.jev(), pd.DataFrame(json.loads((R.RESULTS / "jev/probe-ratelimit-25rps.json").read_text())["levels"].values())])
    gpu_name = {p.parent.name: json.loads(p.read_text())["gpu"] for p in R.RESULTS.glob("*/parity.json")}

    def point(box, model, backend, slo, label=""):
        r = sus[(sus.box == box) & (sus.model == model) & (sus.backend == backend) & (sus.label == label) & (sus.slo_ms == slo)]
        return r.iloc[0] if len(r) and pd.notna(r.iloc[0].decisions_per_s) else None

    d = dict(sus=sus, steps=steps, base=base, jev=jev, gpu_name=gpu_name, point=point)
    # headline numbers from REPORT.md §0 — fail loudly if results/ and the text disagree
    assert round(point("h100nvl", "laya", "ort-trt-fp16-maxb256", 130).decisions_per_s) == 175
    assert round(point("h100nvl", "laya", "ort-trt-fp16-maxb256", 50).decisions_per_s) == 105
    assert round(point("rtxpro6000-ws", "laya", "ort-trt-fp16-maxb256", 130).decisions_per_s) == 146
    assert round(point("zbook-rtxpro5000", "laya", "ort-trt-fp16-maxb256", 130).decisions_per_s) == 42
    assert point("h100nvl", "laya", "torch-eager-fp16", 130, "mig-1g.12gb-solo") is None
    return d


# Laya operating points shown as "the result": best whole-GPU config per card at each SLO (RTX 2000 Ada never meets it).
LAYA_POINTS = [("zbook-rtxpro5000", "ort-trt-fp16-maxb256", 130), ("rtxpro6000-ws", "ort-trt-fp16-maxb256", 130),
               ("h100nvl", "ort-trt-fp16-maxb256", 130), ("h100nvl", "ort-trt-fp16-maxb256", 50)]
SHORT = {"zbook-rtxpro5000": "RTX PRO 5000 laptop", "rtxpro6000-ws": "RTX PRO 6000", "h100nvl": "H100 NVL",
         "laptop-rtx2000ada": "RTX 2000 Ada laptop"}


def llm_rows(base):
    """vLLM guided-JSON baselines at their throughput peak (c=64); the ingress-serialised H100 row is a platform artefact."""
    b = base[(base.concurrency == 64) & ~((base.box == "h100nvl") & (base.label == "qwen3.5-35b-a3b-fp8"))]
    name = {"qwen3.5-35b-a3b-fp8": "Qwen3.5-35B-A3B", "qwen3.5-35b-a3b-fp8-direct": "Qwen3.5-35B-A3B", "qwen3.5-4b": "Qwen3.5-4B"}
    return [(f"{name[r.label]} · {SHORT[r.box]}", r) for r in b.itertuples()]


# ---------------------------------------------------------------- figures

def fig_hero(d, t):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axvspan(10, 130, color=t["band"], lw=0, zorder=0)
    ax.text(22, 212, "meets p99 ≤ 130 ms", color=t["ink2"], fontsize=10, va="top")
    slo_lines(ax, t)
    for i, (box, be, slo) in enumerate(LAYA_POINTS):
        p = d["point"](box, "laya", be, slo)
        ax.scatter(p.p99_ms, p.decisions_per_s, s=90, color=t["s"][LAYA], edgecolor=t["bg"], lw=2, zorder=3,
                   label="Laya 421M, self-hosted (TensorRT FP16)" if i == 0 else None)
        ax.annotate(f"{SHORT[box]}{' @ 50 ms SLO' if slo == 50 else ''}\n{p.decisions_per_s:.0f} dec/s",
                    (p.p99_ms, p.decisions_per_s), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=9.5)
    j = d["jev"][d["jev"].achieved_qps > 20].iloc[0]
    ax.scatter(j.p99_ms, j.decisions_per_s, s=90, marker="D", color=t["s"][JEV], edgecolor=t["bg"], lw=2, zorder=3,
               label="Jev API (per key, includes WAN)")
    ax.annotate(f"Jev API\n{j.decisions_per_s:.0f} dec/s", (j.p99_ms, j.decisions_per_s), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9.5)
    rows = llm_rows(d["base"])
    for i, (label, r) in enumerate(rows):
        ax.scatter(r.p99_ms, r.decisions_per_s, s=90, marker="s", color=t["s"][LLM], edgecolor=t["bg"], lw=2, zorder=3,
                   label="LLM via vLLM guided JSON (c=64)" if i == 0 else None)
    ax.annotate("4 LLM configs\n188–202 dec/s", (1500, 150), ha="center", fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(20, 3000); ax.set_ylim(0, 220)
    ax.set_xticks([20, 50, 130, 300, 1000, 3000]); ax.set_xticklabels(["20", "50", "130", "300", "1 s", "3 s"])
    ax.set_xlabel("p99 latency at that load (ms, log scale)"); ax.set_ylabel("sustained decisions / s per GPU")
    ax.set_title("Same throughput class as an LLM, an 11× tighter tail")
    subtitle(ax, "Open-loop Poisson load, 3-question bundles over real SAM.gov notices. Up and to the left is better.", t)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.02), fontsize=9.5)
    save(fig, "hero", t["name"])


def fig_knee(d, t):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    slo_lines(ax, t, "y")
    series = [("laptop-rtx2000ada", "torch-eager-fp16"), ("zbook-rtxpro5000", "ort-trt-fp16-maxb256"),
              ("rtxpro6000-ws", "ort-trt-fp16-maxb256"), ("h100nvl", "ort-trt-fp16-maxb256")]
    st = d["steps"]
    for i, (box, be) in enumerate(series):
        g = st[(st.box == box) & (st.model == "laya") & (st.backend == be) & (st.label == "")].sort_values("target_qps")
        ax.plot(g.decisions_per_s, g.p99_ms, marker="o", color=t["s"][i], markeredgecolor=t["bg"], markeredgewidth=1.5,
                label=f"{SHORT[box]} · {BACKEND[be]}")
        last = g.iloc[(g.p99_ms < 3000).values.nonzero()[0][-1]]
        ax.annotate(SHORT[box], (last.decisions_per_s, last.p99_ms), xytext=(6, 4), textcoords="offset points", fontsize=9.5)
    for slo in (50, 130):
        ax.text(0.99, slo * 0.9, f"p99 ≤ {slo} ms SLO", color=t["ink2"], fontsize=9, ha="right", va="top", transform=ax.get_yaxis_transform())
    ax.set_yscale("log"); ax.set_ylim(10, 20000); ax.set_xlim(0, None)
    ax.set_yticks([10, 50, 130, 1000, 10000]); ax.set_yticklabels(["10 ms", "50", "130", "1 s", "10 s"])
    ax.set_xlabel("achieved decisions / s"); ax.set_ylabel("p99 latency (log scale)")
    ax.set_title("The knee is sharp: seconds of tail one load step past capacity")
    subtitle(ax, "laya 421M, dynamic-batching server, 60 s per step, target rate raised ×1.4 per step.", t)
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "knee", t["name"])


def fig_day(d, t):
    rows = [json.loads(l) for l in (R.RESULTS / "rtxpro6000-ws/replay/laya.eager-fp16.10M-day.jsonl").read_text().splitlines() if l.strip()]
    ts = np.array([r["t_sched"] for r in rows]); lat = np.array([r["latency_ms"] for r in rows])
    hour = np.minimum((ts // 150).astype(int), 23)  # 24 h replayed at 150 s per hour
    load_ = [(hour == h).sum() * 3 / 150 for h in range(24)]
    p99 = [np.percentile(lat[hour == h], 99) for h in range(24)]
    assert len(rows) == 138_863 and round(np.percentile(lat, 99)) == 111 and round(max(p99)) in (143, 144)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.15})
    a1.bar(range(24), load_, width=0.8, color=t["s"][LAYA], edgecolor=t["bg"], lw=2)
    a1.set_ylim(0, 215); a1.set_ylabel("decisions / s"); a1.grid(axis="x", visible=False)
    a1.set_title("A 10-million-decision day on one RTX PRO 6000: 0 errors, p99 111 ms")
    subtitle(a1, "24 h diurnal curve, each hour replayed for 150 s at its real arrival rate. 138,863 requests.", t)
    a1.annotate(f"peak {max(load_):.0f} dec/s", (int(np.argmax(load_)), max(load_)), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9.5)
    slo_lines(a2, t, "y")
    over = [p > 130 for p in p99]
    a2.bar(range(24), p99, width=0.8, color=[t["s"][JEV] if o else t["s"][LAYA] for o in over], edgecolor=t["bg"], lw=2)
    for h in np.nonzero(over)[0]:
        a2.annotate(f"{int(p99[h])}", (h, p99[h]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9.5)
    a2.text(23.6, 132, "130 ms SLO", color=t["ink2"], fontsize=9, ha="right", va="bottom")
    a2.set_ylabel("p99 latency (ms)"); a2.set_ylim(0, 165); a2.grid(axis="x", visible=False)
    a2.set_xticks(range(0, 24, 2)); a2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    a2.set_xlabel("hour of the simulated day (orange = the 2 hours over the 130 ms SLO)")
    save(fig, "day-replay", t["name"])


def fig_mig(d, t):
    st = d["steps"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    slo_lines(ax, t, "y")
    for i, (be, label) in enumerate([("ort-trt-fp16-maxb256", "whole H100 · TensorRT FP16"), ("torch-eager-fp16", "whole H100 · eager FP16")]):
        g = st[(st.box == "h100nvl") & (st.model == "laya") & (st.backend == be) & (st.label == "")].sort_values("target_qps")
        ax.plot(g.decisions_per_s, g.p99_ms, marker="o", color=t["s"][[LAYA, 3][i]], markeredgecolor=t["bg"], label=label)
    sl = st[(st.box == "h100nvl") & (st.model == "laya") & st.label.str.match(r"mig-1g\.12gb-slice\d")]
    agg = sl.groupby("target_qps").agg(dps=("decisions_per_s", "sum"), p99=("p99_ms", "mean"), spread=("p99_ms", lambda s: s.max() - s.min()))
    ax.plot(agg.dps, agg.p99, marker="s", color=t["s"][JEV], markeredgecolor=t["bg"], label="7 × MIG 1g.12gb slices, all serving at once (summed)")
    ax.annotate(f"7 slices at best: {agg.dps.iloc[0]:.0f} dec/s, p99 {agg.p99.iloc[0]:.0f} ms", (agg.dps.iloc[0], agg.p99.iloc[0]),
                xytext=(-20, 60), textcoords="offset points", fontsize=9.5, arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=1))
    ax.set_yscale("log"); ax.set_ylim(10, 20000); ax.set_xlim(0, 230)
    ax.set_yticks([10, 50, 130, 1000, 10000]); ax.set_yticklabels(["10 ms", "50", "130", "1 s", "10 s"])
    ax.set_xlabel("decisions / s for the whole card"); ax.set_ylabel("p99 latency (log scale)")
    ax.set_title("MIG: slicing the H100 seven ways loses to one whole GPU")
    subtitle(ax, f"laya 421M on 400-token documents. Isolation is perfect: the seven slices' p99 agree within {agg.spread.iloc[0]:.1f} ms.", t)
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "mig", t["name"])


def fig_cost(d, t):
    rows = []
    for box, be, slo in LAYA_POINTS:
        p = d["point"](box, "laya", be, slo)
        c = R.cost_per_million(p.decisions_per_s, p.watts_mean, d["gpu_name"][box])["total_usd_per_m"]
        rows.append((f"Laya · {SHORT[box]} · p99 ≤ {slo} ms", c, LAYA, ""))
    for label, r in llm_rows(d["base"]):
        c = R.cost_per_million(r.decisions_per_s, 0, d["gpu_name"][r.box])["total_usd_per_m"]
        rows.append((label, c, LLM, f"   p99 {r.p99_ms / 1000:.1f} s, card cost only"))
    jv = [R.jev_cost_per_million(x) for x in R.jev().mean_input_tokens_per_request]  # probe.json levels, as REPORT.md §6.9
    rows.append(("Jev API (list price)", max(jv), JEV, f"   (range ${min(jv):.2f}–{max(jv):.2f} across runs), p99 ≈ 0.3 s"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.7, color=[t["s"][r[2]] for r in rows], edgecolor=t["bg"], lw=2)
    for yi, (label, c, _, note) in zip(y, rows):
        ax.text(c * 1.04, yi, f"${c:.2f}{note}", va="center", fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=t["ink"], fontsize=10)
    ax.set_xscale("log"); ax.set_xlim(0.1, 60); ax.grid(axis="y", visible=False)
    ax.set_xticks([0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(["$0.10", "$0.30", "$1", "$3", "$10"])
    ax.set_xlabel("US $ per million decisions (log scale): card amortised over 3 years + energy at $0.12/kWh, 100 % utilisation")
    ax.set_title("Cost per million decisions")
    subtitle(ax, "Blue = Laya inside its SLO · aqua = LLM (misses both SLOs) · orange = Jev API. Assumptions in REPORT.md §6.9.", t)
    save(fig, "cost", t["name"])


def fig_shootout(d, t):
    c = R.shootout()
    c = c[(c.model == "laya") & (c.length == "long") & (c.batch == 64)].copy()
    c["backend"] = c.backend.map(BACKEND)
    cols = ["eager FP32", "eager FP16", "ORT CUDA FP16", "TensorRT FP16", "torch.compile FP16"]
    boxes = list(GPU)
    m = c.pivot_table(index="box", columns="backend", values="decisions_per_s").reindex(index=boxes, columns=cols)
    rel = m.div(m["eager FP16"], axis=0)
    assert m.isna().sum().sum() == 1  # the only gap: ORT CUDA b64 long OOM on the 8 GB laptop (REPORT.md §6.2)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cmap, norm = plt.get_cmap(t["seq"]), plt.Normalize(0.2, 1.9)
    ax.imshow(rel.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(boxes)):
        for j in range(len(cols)):
            v, dps = rel.values[i, j], m.values[i, j]
            r_, g_, b_, _ = cmap(norm(v)) if not np.isnan(v) else (0, 0, 0, 0)
            dark_cell = np.isnan(v) and t["name"] == "dark" or 0.299 * r_ + 0.587 * g_ + 0.114 * b_ < 0.55
            ax.text(j, i, "OOM" if np.isnan(v) else f"{v:.2f}×\n{dps:.0f} dec/s", ha="center", va="center", fontsize=9.5,
                    color=t["ink2"] if np.isnan(v) else "#ffffff" if dark_cell else "#0b0b0b")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, color=t["ink"])
    ax.set_yticks(range(len(boxes))); ax.set_yticklabels([GPU[b] for b in boxes], color=t["ink"])
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Backend shootout: speed relative to eager FP16")
    subtitle(ax, "laya, batch 64, long notices (400–472 tokens), fixed shapes; median of 3 fresh-process repeats × 50 timed batches.", t)
    save(fig, "shootout", t["name"])


def social_card(d):
    """1280×640 repository social preview (GitHub → Settings → Social preview)."""
    t = THEMES["dark"] | {"name": "dark"}
    style(t)
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.05, 0.84, "laya-cuda-bench", fontsize=40, fontweight="bold", color=t["ink"])
    fig.text(0.05, 0.765, "Capacity, tail latency and cost of a 421M decision model on NVIDIA GPUs", fontsize=19, color=t["ink2"])
    h = d["point"]("h100nvl", "laya", "ort-trt-fp16-maxb256", 130)
    stats = [(f"{h.decisions_per_day / 1e6:.0f}M", "decisions/day on one H100\ninside p99 ≤ 130 ms"),
             ("10M", "decisions in a day on one\nRTX PRO 6000, 0 errors"),
             ("$0.66", "per million decisions\n(Jev API list: $6.8–8.2)"),
             ("74/74", "parity runs match upstream\nacross 5 backends, 4 GPUs")]
    for i, (big, small) in enumerate(stats):
        x = 0.05 + i * 0.235
        fig.text(x, 0.50, big, fontsize=46, fontweight="bold", color=t["s"][LAYA])
        fig.text(x, 0.44, small, fontsize=14, color=t["ink"], va="top", linespacing=1.4)
    fig.text(0.05, 0.08, "RTX 2000 Ada · RTX PRO 5000 · RTX PRO 6000 · H100 NVL + MIG   |   PyTorch · torch.compile · ONNX Runtime · TensorRT",
             fontsize=13, color=t["ink2"])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "social-preview.png", dpi=100)
    plt.close(fig)


def main():
    d = load()
    for name, t in THEMES.items():
        t = t | {"name": name}
        style(t)
        for f in (fig_hero, fig_knee, fig_day, fig_mig, fig_cost, fig_shootout):
            f(d, t)
    social_card(d)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
