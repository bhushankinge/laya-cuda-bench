# laya-cuda-bench

Capacity-planning benchmark for [Laya](https://github.com/NandhaKishorM/laya), Convai's open-weights
"System 1" decision models (421M English / 322M multilingual), on NVIDIA GPUs. The question it answers is
the one an engineering team asks before self-hosting a decision model behind every agent call:

> How many typed decisions per second does one GPU sustain under a p99 latency SLO, what does that
> cost per million decisions, and how does it compare with the Jev API and with an LLM doing the same job?

Everything here is reproducible from this repository: pinned sources and weights (`manifest.json`),
frozen public fixtures (`fixtures/`), raw per-request and per-iteration samples (`results/`), and the
notebook that draws every chart from those files (`notebooks/report.ipynb`).

## What is measured

| Section | Method | Files |
|---|---|---|
| **Parity gate** | 63 questions from the laya-mlx port's parity cases; every backend must match the upstream FP32 answer 63/63 and hold zero steady-state allocator growth over 100 calls | `harness/parity.py`, `results/<box>/parity.json` |
| **Server (headline)** | MLPerf-Server-style: open-loop Poisson arrivals against a dynamic-batching server, target QPS raised geometrically, max sustained decisions/s with **p99 ≤ 50 ms** and **p99 ≤ 130 ms** | `harness/server.py`, `harness/loadgen.py`, `harness/sweep.py`, `results/<box>/server/` |
| **Offline** | largest batch that fits, 50 timed batches → decisions/s (comparable to upstream's batched T4 and the port's 50-question M3 Max numbers) | `harness/offline.py` |
| **Large-org day** | a 24 h diurnal curve scaled to 10M decisions/day; each hour is replayed for 150 s at that hour's real arrival rate, so the 1 h run walks the day's shape without inflating the load | `fixtures/workload/diurnal.csv`, `harness/loadgen.py --curve` |
| **Backend shootout** | FP16 eager / `torch.compile` / ONNX Runtime CUDA / ONNX Runtime TensorRT (+ FP32 eager reference), batch {1,16,64,256} × {short,long}, 20 warmup + 50 timed, 3 fresh-process repeats | `harness/shootout.py` |
| **MIG** | H100 NVL: whole GPU vs one 2g.24gb slice vs one 1g.12gb slice vs **7 × 1g.12gb serving at once** (aggregate per card) | `k8s/`, `scripts/h100_window.md` |
| **Baselines** | Jev API (`jev-1.13.0`, same 3 questions, measured over the network), Qwen3.5-35B-A3B-FP8 and Qwen3.5-4B via vLLM guided JSON on the same GPUs | `harness/jev_probe.py`, `harness/llm_baseline.py` |
| **Cost** | $/M decisions = energy at the SLO operating point + amortised card (3 years); $/M input tokens for the Jev row | `harness/report.py` |

### Workload

A request is the 3-question bundle an agent fires per event — `route` (choice, 3 options), `severity`
(score, 3 levels), `needs_human` (noul) — over one of 1,000 public U.S. federal contract-opportunity
notices sampled from SAM.gov's data.gov bulk extract (`fixtures/workload/`, provenance in `source.json`).
Length buckets by each checkpoint's own tokenizer: short ≤ 80, medium 160–360, long 400–472, xlong 700–984 tokens.

### Timing boundary

Shootout and Offline time prompt construction → tokenization → host-to-device copy → forward →
device-to-host copy (which synchronizes) → temperature calibration → answer formatting; model load is
excluded. This is the boundary laya-mlx's BENCHMARKS.md uses, so those rows are comparable. Server
numbers are client-observed request latency over loopback HTTP, including queueing and batching.
Power is board power sampled in-process with NVML every 100 ms (on a MIG slice this is the whole card).

### Batching

Upstream `Agent.predict` batches the questions of one state. Here a batch is any set of (state, question)
rows built with the upstream `build_sequence`, so tokens are identical, and the model is called directly
(`harness/sequences.py`), the same way the MLX port produced its 64-batch throughput number.

## Reproduce

```bash
./setup_env.sh .venv python3.12          # torch 2.14 cu130, laya @ pinned commit, ORT 1.30 (CUDA 13 feed), TensorRT 11.3
hf download convaiinnovations/laya --revision c5d78730f3493e4fe16d61507ef4b78eef7318cf --local-dir models/laya
hf download convaiinnovations/laya-multilingual --local-dir models/laya-multilingual
hf download convaiinnovations/laya-typed-decisions --local-dir models/laya-typed-decisions
python scripts/make_manifest.py          # compare the SHA256s with the committed manifest.json
python -m harness.envinfo --box mybox
python -m harness.export_onnx
python -m harness.parity --box mybox --backends eager-fp32 eager-fp16 ort-cuda-fp16 ort-trt-fp16 compile-fp16
python -m harness.shootout --box mybox
python -m harness.sweep --box mybox --model laya --backend eager-fp16
python -m harness.offline --box mybox --model laya --backend eager-fp16
```

`fixtures/workload/make_workload.py` rebuilds the workload from the current SAM.gov extract; the committed
`states.jsonl` is the frozen version used for every number here.

## Results

See `notebooks/report.ipynb` and the blog post (link added at publication). Per-box raw data lives under
`results/<box>/` with an `env.json` describing driver, CUDA, library versions, clocks and MIG state.

## Limitations

- Parity on 63 fixtures is fidelity to upstream, not task accuracy.
- The Jev row is measured over the public internet from Arizona; its latency includes the network.
- MIG power is per physical card; per-slice energy is not measurable with NVML.
- The load generator runs on the same host over loopback; the serving stack is a ~120-line asyncio
  dynamic batcher, not Triton (same algorithm; Triton is a follow-up row).
- LLM baselines report latency/throughput/tokens only; no accuracy claims are made for any model.

## License

Apache-2.0. Third-party attributions in `NOTICE`.
