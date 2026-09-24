"""Turn results/<config>/qps*.summary.json into one markdown table (rows = offered rate, cols = configs)."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/serve_batching")
configs = [c for c in ("main-970dc8c", "pr434-defaults", "pr434-w10-b16", "pr434-w10-b4") if (R / c).is_dir()]
rates = sorted({float(p.stem[3:-8]) for c in configs for p in (R / c).glob("qps*.summary.json")})
print("| offered q/s | " + " | ".join(configs) + " |")
print("|---|" + "---|" * len(configs))
for q in rates:
    cells = []
    for c in configs:
        p = R / c / ("qps%g.summary.json" % q)
        if not p.exists():
            cells.append("—"); continue
        s = json.loads(p.read_text())
        if not s.get("ok"):
            cells.append("no ok responses"); continue
        cells.append("%.1f q/s, p50 %.0f, p99 %.0f ms%s" % (
            s["achieved_qps"], s["p50_ms"], s["p99_ms"],
            (", %d err" % s["errors"]) if s.get("errors") else "") + (
            ", trend %.1fx" % s["latency_trend"] if s["latency_trend"] > 1.5 else ""))
    print("| %g | " % q + " | ".join(cells) + " |")
