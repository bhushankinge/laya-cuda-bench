---
license: apache-2.0
pretty_name: laya-cuda-bench — GPU capacity, tail latency and cost results for the Laya decision models
language:
  - en
tags:
  - benchmark
  - gpu
  - inference
  - latency
  - tensorrt
  - onnxruntime
  - torch-compile
  - mig
  - h100
  - blackwell
  - laya
size_categories:
  - 100K<n<1M
---

# laya-cuda-bench results

Raw measurements behind **[laya-cuda-bench](https://github.com/bhushankinge/laya-cuda-bench)**, a capacity-planning
study of [Laya](https://huggingface.co/convaiinnovations/laya) (Convai's open-weights 421M English / 322M multilingual
"System 1" typed-decision models) on NVIDIA GPUs: how many decisions per second one GPU sustains under a p99 SLO,
and what a million of them cost.

![Sustained decisions/s against p99 latency for Laya, the Jev API and LLM baselines](https://raw.githubusercontent.com/bhushankinge/laya-cuda-bench/main/docs/figures/hero.png)

| GPU (`laya`, TensorRT FP16) | p99 ≤ 50 ms | p99 ≤ 130 ms | $/M decisions at 130 ms |
|---|---|---|---|
| RTX PRO 5000 Blackwell Laptop 24 GB | 15 dec/s | 42 dec/s | $0.67 |
| RTX PRO 6000 Blackwell 96 GB | — | 146 dec/s | $0.66 |
| H100 NVL 94 GB | 105 dec/s | 175 dec/s | $1.86 |

A 10M-decision diurnal day on one RTX PRO 6000: 0 errors, p99 111 ms. LLM baselines (Qwen3.5-35B-A3B-FP8,
Qwen3.5-4B, vLLM guided JSON) reach ~200 decisions/s per card at p99 ≈ 1 s. Full analysis:
[REPORT.md](https://github.com/bhushankinge/laya-cuda-bench/blob/main/REPORT.md).

## Contents

| Path | Rows | Schema |
|---|---|---|
| `results/<box>/env.json` | 1 per GPU | driver, CUDA, library versions, clocks, MIG state |
| `results/<box>/parity.json` | 1 per GPU | per model × backend: argmax agreement /63, max probability error, allocator growth |
| `results/<box>/shootout/<model>/<backend>.r<k>.jsonl` | 1 per timed batch | `box, gpu, model, backend, batch, length, repeat, iter, ms, n_decisions, seq_len, tokens, peak_vram_mb, watts_mean, sm_mhz_mean, ts` |
| `results/<box>/server/*.qps<N>.jsonl` | 1 per request | `t_sched, t_send, t_done, latency_ms, send_lag_ms, status, served_batch, n_decisions, input_tokens` |
| `results/<box>/server/*.sweep.json` | 1 per sweep | `sustained: {50: …, 130: …}` and per-step p50/p99, achieved rate, served batch, power, saturation |
| `results/<box>/replay/*.jsonl` | 1 per request | same as server rows; 24 h diurnal replay |
| `results/<box>/offline/*.json` | 1 per run | largest batch, decisions/s, watts, J/decision, peak VRAM |
| `results/<box>/baselines/*` | per level / request | LLM baselines via vLLM guided JSON |
| `results/jev/*` | per level / request | hosted Jev API probe (`jev-1.13.0`) |
| `fixtures/workload/states.jsonl` | 1,000 | public SAM.gov contract notices used as inputs (`source.json` has provenance) |
| `fixtures/workload/questions.json`, `diurnal.csv` | — | the 3 typed questions and the 24 h load curve |

Boxes: `laptop-rtx2000ada`, `zbook-rtxpro5000`, `rtxpro6000-ws`, `h100nvl` (whole GPU and `*.mig-1g.12gb*` slices).

## Load it

```python
import pandas as pd
rows = pd.read_json("hf://datasets/bhushankinge/laya-cuda-bench/results/rtxpro6000-ws/replay/laya.eager-fp16.10M-day.jsonl", lines=True)
print(rows.latency_ms.quantile(0.99))  # 111 ms
```

## Caveats

Parity is fidelity to upstream FP32 answers, not task accuracy. Cost assumes 3-year card amortisation at 100 %
utilisation and $0.12/kWh. One run per sweep configuration (shootout: 3 fresh-process repeats). The complete list is
in the report's §10.

## Citation

```bibtex
@misc{kinge2026layacudabench,
  author       = {Kinge, Bhushan},
  title        = {laya-cuda-bench: Capacity, Tail Latency and Cost of the Laya Decision Models on NVIDIA GPUs},
  year         = {2026},
  howpublished = {\url{https://github.com/bhushankinge/laya-cuda-bench}}
}
```
