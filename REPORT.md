# laya-cuda-bench — full study report

**Laya (Convai's open-weights "System 1" decision models) on NVIDIA GPUs: capacity under a p99 SLO, cost per million decisions, and how that compares with the Jev API and with LLMs doing the same job.**

Measurement window: 2026-09-22 to 2026-09-23. Four NVIDIA GPUs (RTX 2000 Ada 8 GB laptop, RTX PRO 5000 Blackwell 24 GB laptop, RTX PRO 6000 Blackwell 96 GB workstation, H100 NVL 94 GB whole and sliced into 7 MIG instances), three Laya checkpoints, five inference backends, one API baseline and two LLM baselines. Every number in this document is computed from the raw files under `results/` in this repository; the file that backs each table is named next to it.

---

## 0. Executive summary

1. **Fidelity.** Every backend and precision tested (PyTorch eager FP32/FP16/BF16, `torch.compile` FP16, ONNX Runtime CUDA FP32/FP16, ONNX Runtime TensorRT FP16) reproduces upstream's answers **63/63** on the 16-case / 63-question parity set, on every GPU, for `laya`, `laya-multilingual` and `laya-typed-decisions`, with zero steady-state allocator growth over 100 calls. 74 parity rows in total, no failures.
2. **Backend.** In static (fixed-shape) throughput `torch.compile(mode="max-autotune")` FP16 is the fastest backend on every card: 1.3–1.7× eager FP16 and 1.1–1.5× TensorRT at batch ≥ 64 (the only exceptions are batch-1/16 short cells on the two laptops, where TensorRT or ORT CUDA edge it). Under dynamic-batch serving the picture splits by architecture: **TensorRT wins on Hopper** (H100: 175 vs 93 decisions/s at p99 ≤ 130 ms) while **eager FP16 ties or beats TensorRT on Blackwell** (RTX PRO 6000: 146 vs 146; ZBook: 56 vs 29 for multilingual). `torch.compile` was not used as a serving backend (recompiles per batch shape); it is the most valuable follow-up.
3. **Capacity under SLO, whole GPUs, natural SAM.gov length mix, `laya`:**

   | GPU | p99 ≤ 50 ms | p99 ≤ 130 ms | offline ceiling (no SLO) |
   |---|---|---|---|
   | RTX 2000 Ada Laptop 8 GB (45 W) | not met | not met (multilingual: 5 dec/s) | 39 dec/s (multilingual 84) |
   | RTX PRO 5000 Blackwell Laptop 24 GB (94 W cap) | 15 dec/s = 1.3M/day | 42 dec/s = 3.6M/day | 85 dec/s (multilingual 185) |
   | RTX PRO 6000 Blackwell 96 GB | not measured below 50 q/s (see §6.3) | 146 dec/s = 12.6M/day | 273 dec/s (multilingual 497) |
   | H100 NVL 94 GB, TensorRT | **105 dec/s = 9.0M/day** | **175 dec/s = 15.1M/day** | 339 dec/s (multilingual 647) |

4. **A "large-org day" of 10 million decisions** (24 h diurnal curve, each hour replayed at its real arrival rate) fits on **one RTX PRO 6000** with zero errors and an overall p99 of 111 ms (two peak hours reached 132 and 143 ms). The RTX PRO 5000 laptop carries a 2M/day curve at p99 83 ms.
5. **MIG.** Slicing the H100 into 7 × 1g.12gb is the wrong shape for this workload: a single slice never meets either SLO on the natural mix (with served batches of ~3 requests the tail is already 126–280 ms at 2.6–5 q/s: the slowest single requests on 1/7 of the card exceed the SLO), all seven slices together absorb the 10M/day curve with zero errors but at p99 285 ms, and isolation is perfect (seven concurrent slices reproduce the solo slice's numbers to within 1 ms). Whole-GPU serving beats sliced serving under both SLOs. MIG 1g is a fit for short-prompt decision traffic, not for 400+-token documents.
6. **Cost** (3-year card amortisation at the sustained SLO rate, $0.12/kWh, list-price assumptions in §6.9): Laya self-hosted costs **$0.5–0.7 per million decisions** on the two Blackwell cards (24 GB laptop, 96 GB workstation) and **$1.9/M** on a whole H100 at the 130 ms SLO; energy is 3–7 % of that. **Jev** at $0.042 per million input tokens and 161–194 input tokens per decision costs **$6.8–8.2 per million decisions**. LLMs (Qwen3.5-35B-A3B-FP8 and Qwen3.5-4B via vLLM guided JSON) reach ~190–200 decisions/s per card but with p99 ≈ 1 s, i.e. they never meet either SLO; their card cost is $0.45–1.6/M.
7. **Jev API** (`jev-1.13.0`, measured from Arizona): p50 152–164 ms, p95 223–240 ms, p99 302–334 ms at concurrency 1–32, and **1,500 requests/min sustained for 60 s with zero 429s**, so the third-party "1,200 req/min" ceiling was not enforced at that level.
8. **Premise corrections** worth stating publicly: upstream already has Colab T4 numbers (the gap is everything beyond a T4, not "Apple-only"); upstream's default on any CUDA GPU with compute capability ≥ 8 is **BF16** autocast, not FP16; BF16 has the largest probability error of all dtypes here (up to 0.016) but still 63/63 argmax.
9. **Engineering findings** that will bite anyone reproducing this: torch 2.14 "eager" silently routes some ops through Triton (`TORCH_DISABLE_NATIVE_JIT=1` restores stock kernels); ONNX Runtime's CUDA EP defaults to TF32 (FP32 rows differ by 2.6e-3 unless `use_tf32: 0`); ORT 1.30's TensorRT EP links `libnvinfer.so.10`, so `tensorrt-cu13` must be pinned to 10.x, and the TRT EP silently falls back to CUDA if the libraries are not preloaded; pure-FP16 TensorRT LayerNorm gives 0.025 probability error, the FP32-LayerNorm fallback brings it to 0.002–0.019.
10. **Completeness.** All planned GPU experiments ran. Three gaps remain: the RTX PRO 6000 sweep predates the downward-search fix so its 50 ms SLO row is missing; the optional single 2g.24gb MIG slice was skipped for time; `torch.compile` was never swept as a server. Details in §11.

---

## 1. Research question and design

**Question.** If an engineering team wires a decision model into every agent call, how many typed decisions per second does one GPU (or one MIG slice) sustain under a p99 latency SLO, what does that cost per million decisions, and how does it compare with (a) calling the Jev API and (b) asking an LLM the same questions with constrained JSON output?

**Why this framing and not a latency matrix.** The original checklist was a static GPU × backend × batch × length latency table headlined by a whole H100. That was rejected on evidence: practitioners serve sub-1B encoders on T4/L4/A10G-class hardware; MLPerf retired BERT-large from the datacenter category in v5.0 (its Server constraint had been p99 ≤ 130 ms); NVIDIA positions MIG 1g slices for exactly this model class; the commercial alternative (Jev) is priced per call volume. A batch-1 H100 latency for a 421M-parameter model is a launch-overhead measurement, not a capacity fact. The study therefore keeps a *reduced* static "shootout" for backend selection and comparability with the published T4 / M3 Max rows, and headlines **decisions per day per dollar under an SLO**.

**Two SLOs.** p99 ≤ 50 ms (an "inline guardrail" budget) and p99 ≤ 130 ms (MLPerf's retired BERT Server constraint; roughly Jev's own p50).

**A decision** is one (state, typed question) pair. A request is the 3-question bundle an agent fires per event; throughput is reported in decisions/s (= 3 × requests/s) so Laya, Jev and the LLMs are comparable.

## 2. What was pinned

| Item | Value |
|---|---|
| Upstream `laya` (PyTorch runtime + weights loader) | `NandhaKishorM/laya` @ `6a5819129eb220570792e417e49723d697efd76f` (package version 0.3.3) |
| `laya-mlx` (parity cases, M3 Max reference rows) | `mizorewww/laya-mlx` @ `0a859518634112655cb97c745dbf04f5191aaf13` |
| `receptron/laya` (second ONNX reference) | @ `6478649e723122ca24bbf5fb69ed1010023c9750` |
| Weights `convaiinnovations/laya` | revision `c5d78730f3493e4fe16d61507ef4b78eef7318cf` (HEAD was `1c5edc17…` on 2026-09-22); `model.safetensors` SHA256 `891102d3…`; multilingual `9d628fd9…`; typed-decisions `4fa56de7…` (full hashes in `manifest.json`) |
| Software | torch 2.14.0+cu130, CUDA 13.0, cuDNN 9.24, transformers 5.17.0, onnxruntime-gpu 1.30.0 (CUDA-13 build from the ORT `onnxruntime-cuda-13` feed), tensorrt-cu13 10.16.1.11, nvidia-ml-py 13.610.43, typesafe-sdk 0.7.1, Python 3.12.3 everywhere |
| Workload | 1,000 public U.S. federal contract-opportunity notices sampled (seed 0) from the SAM.gov data.gov bulk CSV, ETag `d1de70c3…`, last-modified 2026-09-22 03:30 UTC (`fixtures/workload/source.json`, `states.jsonl`) |
| Parity fixtures | laya-mlx's 16 cases / 63 questions (eight languages, empty and long states, conversation lists, mask-token literals, structured criteria, mixed batches, 20 options) with upstream FP32 expected outputs (`fixtures/parity/`) |
| Diurnal curve | 24 hourly weights, trough 0.28 at 03:00, peak 1.15 at 10:00 (`fixtures/workload/diurnal.csv`) |

## 3. Method

### 3.1 Parity gate (`harness/parity.py`)
Every backend/precision must reproduce upstream FP32's argmax on all 63 questions. Reported per row: argmax agreements, max absolute error of the calibrated probabilities, max error of action-softmax probabilities, whether the public JSON answers are identical, `torch.compile` time or ORT session/engine build time, and a stability check: 100 repeated calls, finite logits, identical JSON, and steady-state allocator growth (tolerance 1 MiB after a warm call; every row measured −4,608 bytes, i.e. zero). Tolerances follow the MLX port: FP32 ≤ 1e-4, FP16 ≤ 0.02; rows above 1e-3 are annotated. TensorRT parity is run with the same engine profile used for timing (`ort-trt-fp16-maxb<N>`), because engine tactic selection changes the FP16 error.

### 3.2 Backend shootout (`harness/shootout.py`)
Static forward timing per (checkpoint, backend, batch ∈ {1, 16, 64, 256}, length ∈ {short ≤ 80 tokens, long 400–472 tokens}), 20 warm-up + 50 timed iterations, **one fresh process per (checkpoint, backend, repeat)**, 3 repeats (laptop: 1 repeat, batch ≤ 64, eager-FP16 ×3). Timing boundary = prompt construction → tokenization → host-to-device copy → forward → device-to-host copy (synchronising) → calibration → answer formatting; model load excluded. This is the boundary laya-mlx's BENCHMARKS.md uses. Rows are padded multi-state batches built with upstream `build_sequence`, the model is called directly (`harness/sequences.py`), so tokens are identical to upstream. Board power and SM clock are sampled in-process with NVML every 100 ms. Reported: median of the per-iteration medians across repeats.

Backends: `torch-eager-fp32`, `torch-eager-fp16` (autocast), `torch-eager-bf16` (parity only), `torch-compile-max-autotune-fp16`, `ort-cuda-fp16` (ONNX export via `torch.onnx.export` dynamo path, opset 18, dynamic batch/sequence axes, FP16 conversion with `onnxruntime.transformers.float16`), `ort-trt-fp16-maxb<N>` (TensorRT EP, FP16, LayerNorm kept in FP32, engine profile up to batch N × 512 tokens, engine cache on disk). `TORCH_DISABLE_NATIVE_JIT=1` is set so "eager" means stock aten kernels.

### 3.3 Server sweep, the headline (`harness/server.py`, `harness/loadgen.py`, `harness/sweep.py`)
A ~120-line aiohttp server with Triton's dynamic-batching algorithm (take the first waiting request, collect until `max_batch` rows or `max_delay` = 2 ms, one forward, answer everyone; one GPU worker thread). An **open-loop Poisson** load generator (MLPerf Server style) on the same host over loopback offers a target rate; each step is 20 s warm-up + 60 s measured. The target rate rises geometrically (×1.4; laptop ×1.5) until saturation. Saturation = any errors, or p99 > 4 × the loosest SLO, or achieved < 80 % of offered, or `latency_trend` > 2.0 (median latency of the last third of the step / first third, i.e. queue growth). A **sustained** point requires achieved ≥ 90 % of target, p99 ≤ SLO and trend ≤ 1.5. The reported capacity per SLO is the largest sustained target. Since commit `87417db` the sweep also steps *down* until the tightest SLO is met and adds one refinement step at the geometric mean of the best passing and lowest failing rate; the RTX PRO 6000 sweeps predate that fix (§6.3).

Per-box parameters: laptop max-batch 64 from 1 q/s; RTX PRO 5000 max-batch 16 from 5 q/s (a 64-row batch of long notices is a ~1 s forward on that card, so the tail collapses rather than degrades); RTX PRO 6000 and H100 max-batch 128 from 50 q/s; MIG slices max-batch 64 from 10 q/s. Requests draw from the natural SAM.gov mix (749 of the 1,000 notices are ≤ 472 `laya` tokens: 150 short, 238 medium, 150 long, the rest in between; multilingual's tokenizer admits 955 up to 984 tokens).

### 3.4 Offline ceiling (`harness/offline.py`)
Largest medium-length batch that fits (start 2048, halve on OOM), 10 warm-up + 50 timed batches, eager FP16 → decisions/s and J/decision with no latency bound. Comparable to upstream's batched T4 rows and the MLX port's 50-question rows.

### 3.5 Large-org day replay (`harness/loadgen.py --curve`)
The diurnal curve scaled to a daily volume (10M decisions/day on the workstation and datacenter cards, 2M on the RTX PRO 5000, 1/7 of 10M per MIG slice). Each hour is replayed for 3600/compress seconds **at that hour's real arrival rate** (compress 24 → 150 s per hour, one hour total; compress 48 → 75 s per hour, 30 min), so the run walks the day's shape without inflating the load. Reported: overall and per-hour p50/p99, errors, achieved decisions/s. (An earlier version multiplied the rate by the compression factor; fixed in `81eb422` before any committed replay.)

### 3.6 MIG (H100 NVL only)
GPU 0 of a two-GPU node was reconfigured via the NVIDIA GPU Operator's `custom-mig-config` to 7 × 1g.12gb (`all-1g-bd` in `k8s/mig-configs.yaml`; the 1 × 2g.24gb + 4 × 1g.12gb `mixed-bd` layout is defined but was not run). Each slice was served from its own pod. Experiments: one slice alone (parity + both checkpoints' sweeps), then all seven sweeping simultaneously with the same ladder, then a 30-minute replay per slice. NVML reports 0 W for a MIG device, so per-slice power is unavailable.

### 3.7 Baselines
- **Jev API** (`harness/jev_probe.py`): `POST /v1/systemone`, model `jev-1.13.0`, identical request schema to Laya (lowercase types, dict criteria), same 3 questions over the same notices, closed-loop clients at concurrency 1/8/32 under a 10 rps cap, plus a rate-limit probe at 25 rps for 60 s at concurrency 32. Measured from Arizona over the public internet.
- **LLMs** (`harness/llm_baseline.py`): OpenAI-compatible vLLM endpoints, guided JSON via `response_format: json_schema` (route enum / severity 0–2 / needs_human bool), thinking disabled, `max_tokens` 64, temperature 0, closed loop at concurrency 1/16/64 with 60/120/240 requests. Qwen3.5-35B-A3B-FP8 (existing deployments, vLLM 0.17.1 on the RTX PRO 6000, vLLM 0.24.0 on the H100) and Qwen3.5-4B (temporary vLLM servers on both). Latency/throughput/tokens only; no accuracy claim for any model.

### 3.8 The three questions (`fixtures/workload/questions.json`)
`route` (choice: hardware / software / services — "Which team should own this solicitation?"), `severity` (score, 3 levels — "How time-critical is responding to this notice?"), `needs_human` (noul — "Does this notice require a human reviewer before any automated response is sent?").

## 4. Hardware and software environment (`results/<box>/env.json`)

| Box (`results/…`) | GPU | Compute cap. | VRAM | Driver | Max SM clock | Power limit | Host | Tenancy during timing |
|---|---|---|---|---|---|---|---|---|
| `laptop-rtx2000ada` | NVIDIA RTX 2000 Ada Generation Laptop GPU | 8.9 | 8 GB | 590.48.01 | 3,105 MHz | ~45 W (laptop TGP) | HP ZBook Fury G11, 28 threads, 33 GB, Ubuntu 6.17 kernel | on mains; desktop session |
| `zbook-rtxpro5000` | NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU | 12.0 | 24 GB | 595.71.05 | 3,090 MHz | ~94 W (board cap observed) | HP ZBook Fury G1i 18", 24 threads, 67 GB | **sole tenant** (a first run was contaminated by a 15 GB co-tenant and discarded) |
| `rtxpro6000-ws` | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 12.0 | 96 GB | 595.84 | 3,090 MHz | 600 W | 24 threads, 100 GB | exclusive window; `-lgc 2610` requested (2,600 MHz observed during timing) |
| `h100nvl` | NVIDIA H100 NVL (PCIe, NVLink pair) | 9.0 | 94 GB | 580.95.05 | 1,785 MHz | 400 W | Kubernetes pod (`pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime`), 16 vCPU / 64 GiB limit, OMP 8, node 128 threads / 540 GB, RHEL 8 kernel 4.18 | exclusive GPU 0; clocks pinned 1,785 MHz; second GPU running an unrelated workload |

All boxes: torch 2.14.0+cu130, ORT 1.30.0, tensorrt-cu13 10.16.1.11 (the laptop's `env.json` was collected before ORT was installed; its ORT rows used the same versions with a batch-32 TensorRT profile), `TORCH_DISABLE_NATIVE_JIT=1`. TensorRT profile max batch: laptop 32, RTX PRO 5000 64 (parity) / 256 (timing), RTX PRO 6000 128 (parity, day) / 256 (timing), H100 256, MIG slices 64.

## 5. Timeline

| When (Phoenix) | What |
|---|---|
| 2026-09-22 08:00–09:30 | Harness, fixtures, manifest, serving stack, k8s manifests written; Jev probes; sample decisions |
| 09:30–11:43 | Laptop Phase A (parity 20 rows, reduced shootout, sweeps, offline); corrected re-sweep at 11:58 |
| 10:15–12:10 | RTX PRO 5000 first run, discarded (watchdog restarted a 15 GB co-tenant vLLM at 10:15; TensorRT engine builds failed with cuBLAS init errors) |
| 12:32–17:15 | RTX PRO 5000 clean Phase B, sole tenant: parity 21 rows, full shootout, 4 sweeps, offline, 2M/day replay |
| 20:17–23:33 | RTX PRO 6000 Phase C, exclusive: TRT parity at the 256 profile, full shootout, 4 sweeps, offline, 10M/day replay, Qwen3.5-4B baseline (Qwen3.5-35B baseline taken earlier that day) |
| 2026-09-23 11:33–11:50 | Qwen3.5-35B baselines on the H100 (through the platform ingress and direct to the container) |
| 11:50–15:22 | H100 whole GPU: pod setup, parity 10 rows (compile needed a user-space g++), full shootout, 4 sweeps, offline |
| 15:30–15:35 | Qwen3.5-4B on the H100 |
| 15:39–16:49 | MIG 7 × 1g.12gb: transition (< 1 min), 7 pods, solo sweep, 7 concurrent sweeps, 30-min replay |
| 16:50–17:15 | Restore to whole GPU; cluster verified identical to the pre-window snapshot |

## 6. Results

### 6.1 Parity (`results/<box>/parity.json`)

All 71 rows: 63/63 argmax, action-probability error 0, public answers identical, allocator growth 0 (−4,608 B). Max calibrated-probability error by backend and box:

| Checkpoint / backend | RTX 2000 Ada | RTX PRO 5000 | RTX PRO 6000 | H100 NVL |
|---|---|---|---|---|
| laya eager FP32 | 0 | 0 | 0 | 0 |
| laya eager FP16 | 3.1e-3 | 5.3e-3 | 5.5e-3 | 6.7e-3 |
| laya eager BF16 | 1.46e-2 | 9.0e-3 | **1.64e-2** | — |
| laya ORT CUDA FP32 (TF32 off) | 3.2e-6 | 6.5e-6 | 1.4e-6 | — |
| laya ORT CUDA FP16 | 7.5e-3 | 3.2e-3 | 1.01e-2 | 6.3e-3 |
| laya compile FP16 (max-autotune) | 3.3e-3 (25.9 s compile) | 9.5e-3 (16.4 s) | 6.1e-3 (168.7 s) | 5.6e-3 |
| laya ORT TensorRT FP16 | 1.68e-2 (b32 profile, 71 s build) | 1.15e-2 (b64, 3.5 s) | 5.7e-3 (b128, 52 s) / **1.88e-2** (b256, 52 s) | 1.49e-2 (b256) |
| multilingual eager FP16 | 7.7e-4 | 5.4e-4 | 5.3e-4 | 1.1e-3 |
| multilingual eager BF16 | 4.4e-3 | 6.2e-3 | 6.1e-3 | — |
| multilingual ORT CUDA FP16 | 1.5e-3 | 1.0e-3 | 1.0e-3 | 9.8e-4 |
| multilingual compile FP16 | 6.8e-4 (7.7 s) | 6.7e-4 (7.4 s) | 8.7e-4 (114 s) | 7.3e-4 |
| multilingual ORT TensorRT FP16 | — | 2.4e-3 (b64) | 1.7e-3 (b128 and b256, 43 s) | 1.9e-3 (b256) |
| typed-decisions eager FP16 / BF16 / ORT FP16 / compile / TRT | 1.3e-3 / 6.9e-3 / 2.4e-3 / 1.4e-3 / — | 2.3e-3 / 7.9e-3 / 3.0e-3 / 1.5e-3 / 4.8e-3 | 1.2e-3 / 1.48e-2 / 2.2e-3 / 1.2e-3 / 7.7e-3 | — |

Observations: FP32 paths are exact to ≤ 6.5e-6 once ORT's TF32 default is disabled; FP16 sits at 0.5–10e-3; BF16 (upstream's actual default on cc ≥ 8) is the least precise dtype (up to 0.0164) yet never flips an argmax; TensorRT's error depends on the engine build (0.0057 at the batch-128 profile vs 0.0188 at batch-256 on the same card) and approaches the port's 0.02 gate. `laya-typed-decisions` shares `laya`'s architecture and size, so it was parity-tested but not timed separately. The `torch.compile` cold compile ranges from 7 s to 169 s and is excluded from all timings; TensorRT engine builds take 2–72 s and are cached.

### 6.2 Backend shootout (`results/<box>/shootout/<checkpoint>/<backend>.r<k>.jsonl`)

Median decisions/s (= rows/s; 1 row = 1 state × 1 question). Short = ≤ 80 tokens, long = 400–472 tokens. Power = board watts averaged over the timed iterations.

**RTX 2000 Ada Laptop 8 GB** (1 repeat except eager FP16; batch 256 not run; power-capped at ~45 W in every cell above batch 1):

| checkpoint | backend | short b1 | short b16 | short b64 | long b1 | long b16 | long b64 |
|---|---|---|---|---|---|---|---|
| laya | eager FP16 | 57 (17.7 ms) | 135 (118 ms) | 115 (558 ms) | 27 (36.6 ms) | 32 (508 ms) | 30 (2,161 ms) |
| laya | compile FP16 | 73 | 178 | **197** | 34 | 44 | **50** |
| laya | ORT CUDA FP16 | 113 | 143 | 118 | 32 | 26 | OOM |
| laya | ORT TensorRT FP16 (b32) | **143** | **178** | 118 | 33 | 26 | 25 |
| laya | eager FP32 | 38 | 42 | 41 | 10 | 10 | 10 |
| multilingual | eager FP16 | 117 | 360 | 308 | 75 | 69 | 64 |
| multilingual | compile FP16 | 175 | 432 | **470** | 74 | **100** | **104** |
| multilingual | ORT CUDA FP16 | 163 | 358 | 312 | 67 | 54 | 46 |
| multilingual | ORT TensorRT FP16 (b32) | **192** | **433** | 290 | 60 | 48 | 48 |
| multilingual | eager FP32 | 103 | 121 | 120 | 23 | 26 | 24 |

**RTX PRO 5000 Blackwell Laptop 24 GB** (3 repeats, sole tenant; board hits its ~94 W cap from batch 16 up, so throughput is flat past batch 16):

| checkpoint | backend | short b1 | short b16 | short b64 | short b256 | long b1 | long b16 | long b64 | long b256 |
|---|---|---|---|---|---|---|---|---|---|
| laya | eager FP16 | 119 (8.4 ms) | 344 (46.5 ms) | 330 | 286 (895 ms) | 72 (13.9 ms) | 80 (200 ms) | 72 | 72 (3,566 ms) |
| laya | compile FP16 | **180** | **464** | **478** | **427** | **88** | **112** | 103 | **104** |
| laya | ORT CUDA FP16 | 200 | 289 | 291 | 248 | 62 | 57 | 51 | 48 |
| laya | ORT TensorRT FP16 (b256) | 169 | 458 | 421 | 376 | 71 | 107 | **97** | 96 |
| laya | eager FP32 | 91 | 109 | 117 | 115 | 29 | 29 | 27 | 29 |
| multilingual | eager FP16 | 153 | 867 | 815 | 631 | 144 | 180 | 148 | 143 |
| multilingual | compile FP16 | **331** | **1,241** | **1,323** | **1,055** | **147** | **277** | **231** | **224** |
| multilingual | ORT CUDA FP16 | 299 | 799 | 819 | 581 | 128 | 119 | 97 | 89 |
| multilingual | ORT TensorRT FP16 (b256) | 321 | 1,044 | 1,050 | 785 | 119 | 133 | 115 | 108 |
| multilingual | eager FP32 | 148 | 285 | 341 | 256 | 60 | 63 | 58 | 64 |

**RTX PRO 6000 Blackwell Workstation 96 GB** (3 repeats, exclusive, ~2,600 MHz):

| checkpoint | backend | short b1 | short b16 | short b64 | short b256 | long b1 | long b16 | long b64 | long b256 |
|---|---|---|---|---|---|---|---|---|---|
| laya | eager FP16 | 44 (22.5 ms) | 507 (31.5 ms) | 878 (72.9 ms) | 753 (340 ms) | 41 (24.5 ms) | 252 (63.5 ms) | 216 (296 ms) | 213 (1,203 ms) |
| laya | compile FP16 | **249** | **889** | **1,043** | **1,040** | **148** | **299** | **305** | **301** |
| laya | ORT CUDA FP16 | 127 | 814 | 887 | 730 | 111 | 201 | 171 | 162 |
| laya | ORT TensorRT FP16 (b256) | 231 | 849 | 959 | 889 | 143 | 286 | 260 | 254 |
| laya | eager FP32 | 53 | 354 | 363 | 368 | 50 | 105 | 103 | 96 |
| multilingual | eager FP16 | 53 | 605 | 1,372 | 1,247 | 50 | 419 | 369 | 354 |
| multilingual | compile FP16 | **348** | **1,356** | **1,621** | **1,633** | **200** | **501** | **522** | **516** |
| multilingual | ORT CUDA FP16 | 154 | 1,230 | 1,486 | 1,323 | 133 | 355 | 316 | 287 |
| multilingual | ORT TensorRT FP16 (b256) | 275 | 1,272 | 1,471 | 1,387 | 180 | 331 | 301 | 291 |
| multilingual | eager FP32 | 64 | 693 | 826 | 790 | 62 | 211 | 208 | 210 |

Power at batch 256 short/long: eager FP16 347/397 W, compile 323/384 W, TRT 351/404 W, FP32 510/536 W. Note the anomalous eager FP16 batch-1 cells (22.5 ms, slower than the laptops): with clocks pinned high and stock aten kernels, batch-1 eager on this card is launch-overhead bound; compile and TensorRT remove it (4.0 and 4.3 ms).

**H100 NVL 94 GB** (3 repeats, exclusive, 1,785 MHz pinned, in a container):

| checkpoint | backend | short b1 | short b16 | short b64 | short b256 | long b1 | long b16 | long b64 | long b256 |
|---|---|---|---|---|---|---|---|---|---|
| laya | eager FP16 | 93 (10.7 ms) | 835 (19.2 ms) | 983 (65.1 ms) | 1,002 (256 ms) | 88 (11.4 ms) | 280 (57.1 ms) | 291 (220 ms) | 294 (871 ms) |
| laya | compile FP16 | **294** | **1,292** | **1,451** | **1,493** | **194** | **439** | **460** | **464** |
| laya | ORT CUDA FP16 | 181 | 824 | 916 | 919 | 111 | 213 | 215 | 211 |
| laya | ORT TensorRT FP16 (b256) | 262 | 999 | 1,107 | 1,128 | 134 | 336 | 346 | 341 |
| laya | eager FP32 | 114 | 246 | 257 | 270 | 50 | 71 | 69 | 69 |
| multilingual | eager FP16 | 106 | 1,234 | 1,835 | 1,798 | 102 | 486 | 486 | 479 |
| multilingual | compile FP16 | **392** | **2,142** | **2,790** | **2,647** | **267** | **766** | **748** | **739** |
| multilingual | ORT CUDA FP16 | 221 | 1,386 | 1,851 | 1,741 | 148 | 365 | 339 | 334 |
| multilingual | ORT TensorRT FP16 (b256) | 312 | 1,571 | 1,922 | 1,821 | 167 | 322 | 309 | 301 |
| multilingual | eager FP32 | 131 | 603 | 720 | 706 | 93 | 163 | 163 | 161 |

Power at batch 256 short/long: eager FP16 346/357 W, compile 308/331 W, TRT 324/348 W, FP32 397/392 W (400 W limit).

**Cross-GPU reading of the shootout.**
- `torch.compile` FP16 is the fastest static backend in 54 of the 60 non-FP32 cells across the four cards; the six exceptions are batch-1/16 short cells on the two laptops (TensorRT ×4, ORT CUDA ×1, eager ×1). Its advantage over eager is 1.3–1.7× at batch 256 and 1.3–6.6× at batch 1 (largest on the RTX PRO 6000, where stock eager kernels at batch 1 are launch-bound), and 1.1–1.5× over TensorRT at batch ≥ 64.
- TensorRT beats eager FP16 by 1.1–1.7× on `laya` and on short multilingual inputs at batch ≥ 16 on the three larger cards, but is *slower* than eager (0.63–0.82×) on long multilingual inputs on every card, and on long `laya` inputs on the 8 GB laptop (batch-32 engine profile). ORT CUDA EP is slower than eager for long inputs on all four cards (0.66–0.85×).
- Long (400–472 token) notices cost 3.2–4.7× a short (≤ 80 token) one per decision (compile FP16 at the largest batch), on every card.
- Peak short-input throughput per card (compile FP16, `laya`): 197 / 478 / 1,043 / 1,493 decisions/s for RTX 2000 Ada / RTX PRO 5000 / RTX PRO 6000 / H100 NVL; multilingual 470 / 1,323 / 1,633 / 2,790. The H100 is 1.4–1.7× the RTX PRO 6000 in compiled static throughput at similar or slightly lower board power (308–331 W vs 323–384 W at batch 256).
- FP32 is 3–5× slower than FP16 and draws the card's full power limit; it has no serving role.
- Reference points from the literature: upstream's Colab **T4** rows are 39.5 ms (`laya`) / 32.8 ms (multilingual) for one question and 771 / 337 ms for 50 questions (65 / 148 questions/s), i.e. on short inputs every card here is 2–13× a T4 at batch 1 (17.7 ms eager FP16 on the RTX 2000 Ada, 3.4 ms compiled on the H100, vs 39.5 ms) and 3–22× at batch 50–64, as far as the upstream table's inputs are comparable. laya-mlx's **M3 Max** FP16 rows are 17.75 ms for one `laya` question and 347 ms for 50 (143 q/s); multilingual 125 ms for 50 (402 q/s). The 24 GB Blackwell laptop is ~2–2.3× an M3 Max at batch 64 in eager FP16 and ~3.3× with compile.

### 6.3 Server sweeps (`results/<box>/server/<checkpoint>.<backend>[.<label>].sweep.json` + per-step request JSONL)

Sustained capacity (largest offered rate that met the SLO with ≥ 90 % achieved and no latency growth), natural length mix, decisions/s (M decisions/day, mean board W, J/decision):

| Box | checkpoint, backend | max-batch | p99 ≤ 50 ms | p99 ≤ 130 ms |
|---|---|---|---|---|
| RTX 2000 Ada | laya eager FP16 | 64 | not met | not met (p99 141 ms already at 1 q/s) |
| RTX 2000 Ada | multilingual eager FP16 | 64 | not met | 5 dec/s (0.4M, 16 W, 3.46 J) at 1.5 q/s |
| RTX PRO 5000 | laya eager FP16 | 16 | not met (56 ms at 5 q/s) | 42 dec/s (3.6M, 56 W, 1.35 J) at 13.7 q/s |
| RTX PRO 5000 | laya ORT TensorRT FP16 | 16 | 15 dec/s (1.3M, 31 W, 2.04 J) at 5 q/s | 42 dec/s (3.6M, 50 W, 1.20 J) at 13.7 q/s |
| RTX PRO 5000 | multilingual eager FP16 | 16 | 15 dec/s (1.3M, 30 W, 2.00 J) at 5 q/s | **56 dec/s (4.9M, 54 W, 0.96 J)** at 19.2 q/s |
| RTX PRO 5000 | multilingual ORT TensorRT FP16 | 16 | not met (104 ms at 5 q/s) | 29 dec/s (2.5M, 40 W, 1.36 J) at 9.8 q/s |
| RTX PRO 6000 | laya eager FP16 | 128 | *not tested below 50 q/s* | 146 dec/s (12.6M, 212 W, 1.46 J) at 50 q/s |
| RTX PRO 6000 | laya ORT TensorRT FP16 | 128 | *not tested* (p99 68 ms at 50 q/s) | 146 dec/s (12.6M, 186 W, 1.28 J) at 50 q/s |
| RTX PRO 6000 | multilingual eager FP16 | 128 | *not tested* (p99 82 ms at 50 q/s) | 146 dec/s (12.6M, 170 W, 1.17 J) at 50 q/s |
| RTX PRO 6000 | multilingual ORT TensorRT FP16 | 128 | not met | not met (p99 365 ms at 50 q/s) |
| H100 NVL | laya eager FP16 | 128 | 39 dec/s (3.4M, 152 W, 3.89 J) at 13 q/s | 93 dec/s (8.1M, 181 W, 1.94 J) at 30.2 q/s |
| H100 NVL | laya ORT TensorRT FP16 | 128 | **105 dec/s (9.0M, 193 W, 1.84 J)** at 35.7 q/s | **175 dec/s (15.1M, 236 W, 1.35 J)** at 59.2 q/s |
| H100 NVL | multilingual eager FP16 | 128 | not met | 66 dec/s (5.7M, 149 W, 2.27 J) at 21.6 q/s |
| H100 NVL | multilingual ORT TensorRT FP16 | 128 | not met (52 ms at 9.3 q/s) | 93 dec/s (8.0M, 185 W, 2.00 J) at 30.2 q/s |
| H100 1g.12gb slice (solo or 1 of 7) | laya eager FP16 | 64 | not met | not met (p99 126 ms at 2.6 q/s, but achieved 2.3 < 90 %) |
| H100 1g.12gb slice | multilingual eager FP16 | 64 | not met | not met (p99 130–131 ms at 2.6 q/s) |

Every step of every whole-GPU sweep (offered q/s → achieved q/s, p50, p99 ms, mean served batch, W; "SAT" = saturation rule fired):

*RTX 2000 Ada, laya eager FP16 (factor 1.5):* 1.0→0.8, 68/141, b3.2, 19 W · 1.5→1.5, 64/160, 22 W · 2.2→2.1, 59/197, 24 W · 3.4→3.1, 59/150, 28 W · 5.1→4.8, 62/228, 29 W · 7.6→7.6, 83/748, b4.4, 35 W. A single long request's forward is 80–90 ms on this card, so p99 never gets under 130 ms.
*RTX 2000 Ada, multilingual eager FP16:* 1.0→0.8, 36/95 · 1.5→1.5, 34/110, 16 W · 2.2→2.1, 41/131, 19 W · 3.4→3.1, 40/186 · 5.1→4.8, 39/200, 23 W · 7.6→7.6, 42/368, 30 W · 11.4→11.2, 78/1,535, b8.7, 37 W.
*RTX PRO 5000, laya eager FP16 (max-batch 16):* 5.0→5.0, 24.5/55.6, b3.2, 34 W · 7.0→6.7, 23.6/61.4, 39 W · 9.8→9.8, 23.8/72.8, 47 W · 13.7→13.9, 25.0/118.2, 56 W · 19.2→18.8, 29.2/143.1, 67 W · 26.9→27.8, 52.5/947.8, b8.0, 84 W SAT.
*RTX PRO 5000, laya TensorRT FP16:* 5.0→5.0, 18.9/45.3, 31 W · 7.0→6.7, 19.0/60.6, 34 W · 9.8→9.8, 19.7/61.1, 41 W · 13.7→13.9, 20.1/113.1, 50 W · 19.2→18.8, 22.8/135.3, 61 W · 26.9→27.8, 30.9/628.5, 76 W SAT.
*RTX PRO 5000, multilingual eager FP16:* 5.0→5.0, 19.2/44.6, 30 W · 7.0→6.7, 16.8/54.2 · 9.8→9.8, 17.2/50.3, 37 W · 13.7→13.9, 18.1/116.9, 46 W · 19.2→18.8, 18.9/69.8, 54 W · 26.9→27.8, 20.8/153.4, 68 W · 37.6→33.9, 3,901/5,918, b18, 94 W SAT.
*RTX PRO 5000, multilingual TensorRT FP16:* 5.0→5.0, 14.1/103.7, 31 W · 7.0→6.7, 14.2/110.1 · 9.8→9.8, 14.5/121.1, 40 W · 13.7→13.9, 18.5/389.8, 50 W · 19.2→18.8, 21.5/532.2, 62 W SAT.
*RTX PRO 6000, laya eager FP16 (max-batch 128):* 50→48.5, 50.5/86.5, b7.5, 212 W · 70→69.6, 61.6/194.8, b12.4, 294 W · 98→62.9, 12,312/17,193, b129, 387 W, 1,046 errors SAT.
*RTX PRO 6000, laya TensorRT FP16:* 50→48.6, 19.6/67.5, b4.8, 186 W · 70→69.6, 29.5/176.6, b8.2, 261 W · 98→88.6, 4,281/6,661, b129, 448 W SAT.
*RTX PRO 6000, multilingual eager FP16:* 50→48.5, 40.2/82.4, b6.6, 170 W · 70→69.6, 49.3/200.9, b10.9, 236 W · 98→57.8, 14,154/17,687, 347 W, 1,402 errors SAT.
*RTX PRO 6000, multilingual TensorRT FP16:* 50→48.6, 21.3/365.4, b6.7, 198 W · 70→38.9, 19,921/26,151, 393 W, 920 errors SAT.
*H100, laya eager FP16 (max-batch 128, with downward extension and refinement):* 50→48.6, 100.6/199.6, b13, 244 W · 70→69.6, 106.3/327.7, b19, 298 W · 98→97.3, 401/842, b82, 371 W SAT · 35.7→34.9, 22.1/135.3, b4.9, 202 W · 25.5→25.4, 18.7/124.0, 184 W · 18.2→19.6, 18.1/95.0, 174 W · 13.0→13.1, 17.8/38.8, 152 W · 15.4→15.9, 17.9/88.7, 159 W · 30.2→31.1, 19.3/100.3, 181 W.
*H100, laya TensorRT FP16:* 50→48.6, 17.0/63.7, b4.6, 207 W · 70→69.6, 23.8/135.9, b7.2, 266 W · 98→97.7, 170/707, b41, 356 W SAT · 35.7→34.9, 14.5/46.1, 193 W · 42.3→43.7, 15.8/52.6, 206 W · 59.2→58.2, 18.8/91.1, 236 W.
*H100, multilingual eager FP16:* 50→48.5, 119/213, b15, 232 W · 70→69.5, 143/301, b24, 288 W · 98→97.0, 739/1,199, b123, 375 W SAT · 35.7→34.9, 58.8/198 · 25.5→25.4, 18.1/158 · 18.2→19.6, 16.4/107, 152 W · 13.0→13.1, 16.1/95.7, 142 W · 21.6→21.8, 16.5/125.5, 149 W.
*H100, multilingual TensorRT FP16:* 50→43.7, 4,298/7,424, b129, 354 W SAT · 35.7→35.5, 16.7/150.6, 202 W · 25.5→25.4, 15.2/114.9, 183 W · 18.2→18.0, 13.0/66.3 · 13.0→13.8, 12.3/54.8 · 9.3→9.4, 11.8/52.0, 145 W · 30.2→30.9, 14.8/91.8, 185 W · 50 (refinement)→48.1, 41.1/3,150, b23 SAT.

**What the sweeps say.**
- **The knee is sharp.** On every card the tail goes from ~100 ms to seconds within one ×1.4 step, because the batcher fills to `max_batch` once arrivals outrun the forward, and a 64–128-row batch of long notices is a 0.9–1.2 s forward. Capacity planning must sit below the knee, not at "GPU 100 %".
- **Low-rate p99 is set by sequence length, not load.** At 1–15 q/s the served batch is ~3 (the three questions of one request) and p99 equals the forward time of one long notice: 140 ms on the RTX 2000 Ada, 45–60 ms on the RTX PRO 5000, 39–47 ms on the H100 (TensorRT: 46 ms). The 50 ms SLO is therefore mostly a *single-request latency* test on the natural mix, and only TensorRT on the H100 (105 dec/s) and the two laptops at ~5 q/s pass it.
- **Hopper vs Blackwell under serving.** On the H100 eager FP16 is 2× slower under dynamic batching than on the RTX PRO 6000 at the same offered load (p50 100 vs 51 ms at 50 q/s), although its static forward at batch 16 is *faster* (57 vs 64 ms for long inputs). TensorRT shows no such gap (p50 17 vs 20 ms). Hypotheses: the H100 NVL's 1,785 MHz clock makes the many small eager kernels of a 421M model launch-bound at the ~13-row batches the server produces, while the fused TensorRT engine is not; and/or CPU-side batch construction inside the 16-vCPU container. A run with `LAYA_SERVER_DEBUG=1` (per-batch build/forward/post timings) would split this and is listed in §11. Consequence for the recommendation: on Hopper serve with TensorRT (or `torch.compile`); on Blackwell eager FP16 is fine and TensorRT adds nothing under load.
- **TensorRT's dynamic-shape pathology.** Multilingual TensorRT collapsed at 50 q/s on both big cards (p50 4–20 s at batch 128) while passing at 30–36 q/s: the engine re-plans for unseen (batch, sequence) shapes under load. `laya` TRT did not show it. Fixed-shape padding buckets would remove it.
- **RTX PRO 6000 vs H100 at the 130 ms SLO:** 146 vs 175 decisions/s (TensorRT). The workstation card runs at 2,600 MHz and 212 W; the H100 at 1,785 MHz and 236 W. On this model the two are within 20 % of each other; the H100's advantage is memory capacity and MIG, neither of which this workload uses.

### 6.4 Offline ceiling (`results/<box>/offline/*.json`, eager FP16, medium-length inputs, largest batch that fits, 50 timed batches)

| GPU | laya batch | laya dec/s | W | J/decision | multilingual batch | dec/s | W | J/decision |
|---|---|---|---|---|---|---|---|---|
| RTX 2000 Ada (256 OOM) | 128 | 39 | 45 | 1.13 | 256 | 84 | 44 | 0.52 |
| RTX PRO 5000 | 1,024 | 85 | 93 | 1.09 | 1,024 | 185 | 92 | 0.50 |
| RTX PRO 6000 | 2,048 | 273 | 412 | 1.51 | 2,048 | 497 | 347 | 0.70 |
| H100 NVL | 2,048 | 339 | 353 | 1.04 | 2,048 | 647 | 330 | 0.51 |

Offline is 1.9–2.6× the 130 ms-SLO capacity on the big cards: the price of a tail bound on this length mix is roughly half the card. Energy per decision at the ceiling is remarkably flat across four generations (0.50–0.70 J multilingual, 1.04–1.51 J `laya`); the RTX PRO 6000 at 2.6 GHz is the least efficient, the power-capped laptops and the 1,785 MHz H100 the most. Peak VRAM at batch 2048: 34 GB (`laya`), 23 GB (multilingual); at batch 1,024 on the 24 GB laptop, 18 GB.

### 6.5 Large-org day replays (`results/<box>/replay/*.jsonl`)

**RTX PRO 5000, `laya` eager FP16, 2M decisions/day, 24× compressed (1 h), max-batch 16:** 27,870 requests, **0 errors**, p50 24 ms, p90 40 ms, **p99 83 ms**, max 702 ms, 23.2 decisions/s average, 40 at the 10:00 peak. Per-hour p99 stayed between 55 and 102 ms (peak hour 10: 102 ms; hours 8–17 all ≤ 95 ms).

**RTX PRO 6000, `laya` eager FP16, 10M decisions/day, 24× compressed (1 h), max-batch 128:** 138,863 requests, **0 errors**, p50 47 ms, p90 70 ms, **p99 111 ms**, max 251 ms, mean served batch 7.3, 115.7 decisions/s average and 187 at the peak hour. Per-hour p99 (ms): 66, 65, 64, 65, 65, 65, 72, 86, 93, 125, **132**, 126, 111, 126, **143**, 109, 116, 90, 86, 80, 78, 70, 68, 66. Two of the 24 hours (10:00 at 187 dec/s and 14:00 at 185 dec/s) exceed the 130 ms SLO, consistent with the sweep's knee between 146 and 200 decisions/s. One card carries the day; a 130 ms SLO in the two busiest hours needs ~25 % headroom or the TensorRT/compile path.

**H100 NVL, 7 × 1g.12gb slices, `laya` eager FP16, 1/7 of 10M/day per slice, 48× compressed (30 min), max-batch 64:** 7 × 10,017 requests = 70,119, **0 errors**, pooled p50 51 ms, p90 127 ms, **p99 285 ms**, max 942 ms, 16.7 decisions/s per slice (117 for the card), mean served batch 3.5, latency trend 0.9 (no queue growth). Per-hour p99 per slice ranged 125–409 ms; all seven slices are within ±3 ms of each other at every hour. The sliced card takes the volume but cannot hold a 130 ms tail on this length mix at any hour.

### 6.6 MIG on the H100 NVL (`results/h100nvl/server/*mig*`, `results/h100nvl/parity.mig-1g.12gb.json`, `results/h100nvl/WINDOW.md`)

- **Layout run:** GPU 0 → 7 × 1g.12gb (each slice: 1/7 of the SMs, 11.5 GB). The label transition took under a minute; the device plugin advertised the slices about a minute later; the node returned to whole in under a minute at restore. Seven pods were provisioned in 4 minutes by copying one pod's pip cache into the others.
- **Parity on a slice:** `laya` eager FP16 63/63, p_err 6.7e-3 (`parity.mig-1g.12gb.json`).
- **Single slice, alone (`*.mig-1g.12gb-solo.sweep.json`):** `laya`: 2.6 q/s → p50 47 / p99 126 ms; 3.6 → 48 / 191; 5.1 → 48 / 278; 7.1 → 49 / 236; 10 → 92 / 360; 14 → 113 / 538 (SAT). Multilingual: 2.6 → 30 / 131; 3.6 → 32 / 210; 5.1 → 33 / 179; 7.1 → 39 / 218; 10 → 89 / 390; 14 → 89 / 831 (SAT). Neither SLO is met at any offered rate. At 2.6 q/s `laya`'s p99 (126 ms) is under 130 ms but the achieved rate (2.3 q/s) fell below the 90 % gate — Poisson noise with 156 requests per step — so a generous reading is ≈ 2.5 q/s ≈ 7.5 decisions/s per slice at p99 ≈ 130 ms, ≈ 50 decisions/s for all seven slices, versus 93 (eager) / 175 (TensorRT) for the whole card. The p50 at low load (47 ms `laya`, 30 ms multilingual) is the forward time of a typical request on 1/7 of the card; the p99 (126–280 ms at 2.6–5 q/s) is the forward of the longest notices.
- **Seven slices concurrently (`*-slice0..6`):** every step of every slice matches the solo run to within 1 ms of p50 and a few ms of p99 (`laya` at 2.6 q/s: p99 126.1–127.2 ms across the seven; multilingual: 130.4–131.5). MIG isolation is complete for this workload; there is no cross-slice interference to account for.
- **Verdict.** For 400-token documents, one whole GPU with TensorRT or `torch.compile` beats 7 slices of the same GPU under any p99 SLO, by roughly 2–3.5× in sustained decisions/s, and the sliced card's day-replay p99 (285 ms) is 2.6× the whole workstation card's (111 ms). MIG 1g.12gb slices become attractive when inputs are short (a short forward is ~10 ms on a slice) or when the SLO is loose and the goal is bin-packing many low-rate tenants at 11.5 GB each. The 2g.24gb profile (2/7 of the card) was not measured.
- Power per slice is not observable (NVML returns 0 W on a MIG device); the card-level reading is the only energy figure.

### 6.7 Baselines

**Jev API (`results/jev/`, model `jev-1.13.0`, 2026-09-22 16:11–16:15 UTC, from Arizona):**

| clients | requests | ok | achieved req/s | decisions/s | p50 | p95 | p99 | max | input tokens/request |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 100 | 6.1 | 18.4 | 151.5 ms | 222.7 ms | 334.2 ms | 432 ms | 484 |
| 8 (10 rps cap) | 200 | 200 | 10.0 | 29.9 | 157.1 ms | 224.5 ms | 301.6 ms | 310 ms | 532 |
| 32 (10 rps cap) | 300 | 300 | 10.0 | 30.0 | 164.1 ms | 239.4 ms | 329.0 ms | 370 ms | 583 |
| 32 (25 rps, 60 s) | 1,500 | 1,500 | 24.9 | 74.8 | 158.8 ms | 240.0 ms | 310.9 ms | 1,634 ms | 658 |

Zero 429s at 1,500 requests/min. Reported `output_tokens` ≈ 70 per request, billed at $0 (only input tokens are priced, $0.042/M). Jev counts the state once per request (161–194 input tokens per decision on this workload); Laya's bundle feeds the state once per question (the server's replay rows average 789 input tokens per request = 263 per decision), so token counts are not comparable across the two, decisions are. The latency includes an unmeasured WAN round trip.

**LLMs via vLLM, guided JSON, thinking off (`results/<box>/baselines/`), same three questions, ~205–280 prompt tokens, 27 completion tokens:**

| Endpoint | c=1 (60 req) p50 / p99 / dec/s | c=16 (120 req) | c=64 (240 req) |
|---|---|---|---|
| Qwen3.5-35B-A3B-FP8, RTX PRO 6000, vLLM 0.17.1 | 269 ms / 19.2 s (first-call warm-up) / 4.9 | 449 ms / 1.37 s / 78 | 946 ms / 1.31 s / **188** |
| Qwen3.5-4B, RTX PRO 6000, vLLM 0.17.1 | 219 ms / 11.4 s (warm-up) / 7.3 | 373 ms / 471 ms / 118 | 871 ms / 1.09 s / **202** |
| Qwen3.5-35B-A3B-FP8, H100 NVL, vLLM 0.24.0, direct to container | 149 ms / 157 ms / 20 | 599 ms / 701 ms / 74 | 875 ms / 1.06 s / **200** |
| Qwen3.5-35B-A3B-FP8, H100 NVL, through the platform's serving ingress | 156 ms / 203 ms / 17.7 | 3,200 ms / 3,204 ms / 15 | 12,800 ms / 12,810 ms / 15 |
| Qwen3.5-4B, H100 NVL, vLLM 0.24.0 | 130 ms / 1.89 s / 18.8 | 509 ms / 1.38 s / 69 | 902 ms / 987 ms / **200** |

Reading: with guided JSON and 27-token answers both LLMs top out at ~190–200 decisions/s per card regardless of size (4B dense vs 35B MoE with 3B active) and regardless of card (RTX PRO 6000 vs H100): the workload is decode-step bound, not weight bound. Their p99 is 1.0–1.4 s at that throughput and 150–470 ms even at c=1–16, so **no LLM configuration meets either SLO**. Laya on the same H100 sustains 175 decisions/s *inside* a 130 ms p99 (TensorRT) and 105 inside 50 ms. The "through the ingress" row is a platform artifact (requests serialised one at a time by the serving path, 5 req/s flat), kept as a cautionary data point: measure the model, not the gateway.

### 6.8 Sample decisions (`results/samples/laya-vs-jev.jsonl`, 6 notices)
Both models route all six services-type notices (noon meals, Raymarine overhaul, chiller replacement, janitorial, kitchen-hood maintenance, blinds installation) to `services`. Laya's route confidence is 0.48–0.78 versus Jev's 0.80–1.00; Laya's severity scores 0.6–1.2 versus Jev's 0.03–0.36. This is a qualitative sanity check that both APIs answer the same schema, not an accuracy comparison (no labelled set).

### 6.9 Cost per million decisions

Assumptions (all overridable in `harness/report.py`): electricity $0.12/kWh; card street prices September 2026 — RTX 2000 Ada Laptop GPU $650, RTX PRO 5000 Blackwell Laptop $2,500, RTX PRO 6000 Blackwell $8,500, H100 NVL $30,000 per card; 3-year straight-line amortisation; the card runs 24/7 at exactly its sustained SLO rate (100 % utilisation at the knee — real fleets run 30–60 %, which scales the card term by 1.7–3×); host, cooling, network and staff excluded; Jev at $0.042 per million input tokens with the measured 161–194 input tokens per decision.

| Configuration | dec/s | M dec/day | J/dec | energy $/M | card $/M | **total $/M** |
|---|---|---|---|---|---|---|
| RTX PRO 5000, multilingual eager FP16, p99 ≤ 130 | 56 | 4.9 | 0.96 | 0.03 | 0.47 | **0.50** |
| RTX PRO 5000, laya TensorRT FP16, p99 ≤ 130 | 42 | 3.6 | 1.20 | 0.04 | 0.63 | **0.67** |
| RTX PRO 5000, laya TensorRT FP16, p99 ≤ 50 | 15 | 1.3 | 2.04 | 0.07 | 1.76 | **1.83** |
| RTX PRO 6000, laya TensorRT FP16, p99 ≤ 130 | 146 | 12.6 | 1.28 | 0.04 | 0.62 | **0.66** |
| RTX PRO 6000, multilingual eager FP16, p99 ≤ 130 | 146 | 12.6 | 1.17 | 0.04 | 0.62 | **0.66** |
| H100 NVL, laya TensorRT FP16, p99 ≤ 130 | 175 | 15.1 | 1.35 | 0.05 | 1.82 | **1.86** |
| H100 NVL, laya TensorRT FP16, p99 ≤ 50 | 105 | 9.0 | 1.84 | 0.06 | 3.03 | **3.09** |
| H100 NVL, laya eager FP16, p99 ≤ 130 | 93 | 8.1 | 1.94 | 0.07 | 3.40 | **3.46** |
| RTX 2000 Ada, multilingual eager FP16, p99 ≤ 130 | 5 | 0.4 | 3.46 | 0.12 | 1.51 | **1.62** |
| RTX PRO 6000, offline ceiling, no SLO (laya / multilingual) | 273 / 497 | 23.5 / 42.9 | 1.51 / 0.70 | 0.05 / 0.02 | 0.33 / 0.18 | **0.38 / 0.21** |
| H100 NVL, offline ceiling, no SLO (laya / multilingual) | 339 / 647 | 29.3 / 55.9 | 1.04 / 0.51 | 0.03 / 0.02 | 0.94 / 0.49 | **0.97 / 0.51** |
| Qwen3.5-35B-A3B-FP8 or Qwen3.5-4B, RTX PRO 6000, p99 ≈ 1.1–1.3 s | 188–202 | 16.3–17.5 | n/a | n/a | 0.44–0.48 | **≥ 0.45** (+ energy at ~400–600 W: ~$0.07–0.10) |
| Qwen3.5-35B-A3B-FP8 or Qwen3.5-4B, H100 NVL, p99 ≈ 1 s | 200 | 17.3 | n/a | n/a | 1.58 | **≥ 1.6** |
| **Jev API**, p99 ≈ 0.3 s incl. WAN | 75 sustained per key (25 rps) | 6.5 per key | — | — | — | **6.8–8.2** |

Reading: energy is 3–7 % of the self-hosted cost; the card term dominates, so utilisation drives everything. At full utilisation a Blackwell workstation card serves decisions under a 130 ms SLO for ~$0.66/M, about 11× cheaper than Jev's list price and at a p99 2.5–4× lower — but that comparison assumes 12.6M decisions/day of demand to soak the card; below ~1M decisions/day (≈ $7–8/day on Jev) the API wins on price and on zero operations. The H100 is 2.8× the workstation card's cost per decision at the 130 ms SLO because it costs 3.5× as much and is only 1.2× faster on this model. LLMs are cost-competitive per decision at the card level (their throughput is similar) but fail both SLOs by 8–20×.

## 7. Synthesis and recommendations

1. **Right-size the GPU.** A 421M/322M decision model saturates at ~150–175 decisions/s under a 130 ms p99 on both the RTX PRO 6000 and the H100, and at ~1,000–2,800 decisions/s offline. One workstation-class Blackwell card is a 10M-decisions/day machine; an H100 buys ~20 % more at 3.5× the price. Below ~4M/day a 24 GB laptop-class card suffices; below ~1M/day the API is cheaper.
2. **Backend per architecture.** Static: `torch.compile` FP16 everywhere. Serving: TensorRT FP16 on Hopper (and make the profile match the batcher's `max_batch`); eager FP16 on Blackwell unless `torch.compile` with shape buckets is implemented, which the static numbers say would add ~1.5×. ONNX Runtime CUDA EP is never the best choice. FP32 has no role. BF16 (upstream's default) is safe on argmax but the least precise dtype; FP16 is preferable on every card tested.
3. **Set `max_batch` from the SLO, not the memory.** The tail collapses when a batch of long inputs takes longer than the SLO: 16 rows on the 24 GB laptop, 128 rows on the big cards was already too many at the knee. A batch cap of ~SLO / (forward time per long row) keeps the collapse away.
4. **Length is the hidden variable.** Long notices are 3–4× the cost of short ones and set the low-load p99. Any capacity claim must state the length mix; per-bucket reporting is a follow-up (§11).
5. **MIG: not for documents.** Slice only for short-prompt, low-rate, many-tenant traffic. Whole GPU for anything with 400+-token inputs and a tail SLO.
6. **LLMs are not in the same latency class.** ~200 decisions/s with guided JSON at p99 ≈ 1 s versus Laya's 175 at p99 ≤ 130 ms on the same card; and the LLM cost is similar per decision only if you accept the tail.
7. **Jev is the right answer under ~1M decisions/day** and delivers p99 ≈ 0.3 s including WAN with generous rate limits (1,500 req/min observed with no throttling).

## 8. Premise corrections and claims to state carefully

1. Upstream *has* NVIDIA numbers (Colab T4: 32.8 ms single question, 103–332 questions/s batched). What was missing is everything above a T4, batch scaling beyond 50, serving under an SLO, MIG, power and cost.
2. laya-mlx has no ONNX exporter; export from PyTorch (this repo's `harness/export_onnx.py`, or receptron/laya's exporter / `receptron/laya-onnx` on HF as a second reference).
3. Upstream `Agent.predict` batches the questions of one state; multi-state batching means padded rows via `build_sequence` and calling `agent.model` directly, which is what the MLX port did for its batch-64 number and what this server does.
4. Upstream's default autocast on CUDA cc ≥ 8 is **BF16** (`amp_dtype: bf16`), FP16 only on T4-class. So upstream's T4 numbers are FP16 while its default on every GPU here is BF16, which we found to have the largest probability error (up to 0.0164) at 63/63 argmax.
5. `nvidia-smi dmon` floors at 1 s; power here is sampled in-process with NVML at 100 ms. MIG power is per physical card only.
6. Jev's request schema is identical to Laya's; Jev accepted 1,500 requests/min without 429s.
7. Parity on 63 fixtures is *fidelity to upstream*, not task accuracy; no accuracy claim is made for any model, and the six sample decisions are not an evaluation.

## 9. Dead ends and engineering notes (for anyone reproducing or porting)

| Tried | Failed because | Fix in place |
|---|---|---|
| `pip install onnxruntime-gpu` from the ORT CUDA-13 feed with deps | the feed lacks flatbuffers etc. | deps from PyPI first, then `--no-deps` from the feed (`setup_env.sh`) |
| tensorrt-cu13 11.3 | ORT 1.30's TensorRT EP links `libnvinfer.so.10` | pin `tensorrt-cu13==10.16.1.11`; ctypes-preload `tensorrt_libs` |
| Trusting ORT's provider list | the TRT EP silently fell back to the CUDA EP and the row read "PASS" | assert `get_providers()[0]` is the requested EP; backend names carry the profile (`ort-trt-fp16-maxb256`) |
| ORT FP32 as "true FP32" | CUDA EP defaults to TF32 → probability error 2.6e-3 | `use_tf32: 0` for FP32 sessions → 3e-6 |
| ORT arena with the torch model resident on 8 GB | OOM | park the torch model on CPU for ORT backends; `arena_extend_strategy kSameAsRequested` |
| TensorRT profile 256 × 512 on 8 GB / 24 GB | engine build needs ~10 GB workspace | `LAYA_TRT_MAX_BATCH` per box (32 / 64 / 128–256 / 256) |
| TensorRT FP16 default | max-prob error 0.025 | `trt_layer_norm_fp32_fallback` → 0.002–0.019 |
| torch 2.14 "eager" | `torch._native` routes eager ops (e.g. the rotary bmm) through Triton; crashes on boxes without Python headers | `TORCH_DISABLE_NATIVE_JIT=1` in `harness/__init__.py` |
| Memory gate `growth == 0` | the last batch tensor was still referenced | warm call first, `del`, 1 MiB tolerance |
| Parity JSON written at the end | one crash lost every row | write after every backend; re-runs merge into the existing file |
| rsync `results/` to a remote box and back | overwrote fresh laptop results with a stale copy; the H100 slice pods' pull overwrote the whole-GPU `parity.json`/`env.json` (recovered from git) | push code/fixtures/models only; pull only the remote's own subfolder; slice copies saved under `*.mig-1g.12gb.json` |
| `pkill -f` from a waiter script | matched and killed the waiter itself | bracket patterns; chains run detached (`setsid nohup`) |
| Laptop shootout at 3 repeats × batch 256 | ~7 h | reduced laptop scope |
| Saturation = achieved < 0.95 × target | fired at 1 q/s from span noise | rule: errors, or p99 > 4 × loosest SLO, or achieved < 0.8 ×, or latency trend > 2.0; sustained needs trend ≤ 1.5 and achieved ≥ 0.9 × |
| Sweep only stepping down on saturation | the 50 q/s start on the RTX PRO 6000 was already above the 50 ms knee → no 50 ms row | sweep extends downward until the tightest SLO is met and refines each knee (`87417db`); RTX PRO 6000 predates the fix |
| Jev request with capitalised types | 400 Invalid request | lowercase types (identical to Laya's schema) |
| Replay `--total-decisions 10M --compress 24` | rate was multiplied by the compression → 24× the real load | each hour replayed at its real rate for 3600/compress s (`81eb422`) |
| Sweep starting at 20 q/s with max-batch 64 on the 24 GB laptop | first step already past the knee; 64 long rows = 1 s forward | max-batch 16, start 5 q/s |
| Co-tenant GPU on the 24 GB laptop | a watchdog restarted a 15 GB vLLM at the start of the run; TRT engine builds died with `CUBLAS failure 3` | stop the watchdog and containers first; rerun sole-tenant; co-tenant results discarded |
| `pip` inside `pytorch/pytorch:2.14` | PEP 668 (Debian-managed Python) | `PIP_BREAK_SYSTEM_PACKAGES=1` |
| `torch.compile` inside the runtime image | no `g++` (gcc-13 and python3.12-dev present), pod runs as a random non-root UID so no `apt` | `gxx` stage: fetch `g++-13`, `g++-13-x86-64-linux-gnu`, `libstdc++-13-dev` .debs, `dpkg -x` into /tmp, symlink the gcc-13 driver pieces, wrap `g++` with the right `-B`/include paths |
| `curl` in the pod | not in the image | Python `urllib` health checks |
| Seven pods installing from PyPI simultaneously | 17 min for one setup | `PIP_CACHE_DIR` on the first pod, copied into the slice pods: 4 min for seven |
| Namespace limits quota (40 CPU / 100 GiB) | seven pods at 16 CPU / 64 GiB would be refused | slice pods 5 CPU / 13 GiB limits |
| Measuring an LLM through a serving gateway | the gateway serialised requests (5 req/s flat) | port-forward straight to the model container |

## 10. Limitations

- Parity on 63 fixtures is fidelity to upstream, not accuracy; no labelled evaluation was run.
- One run per configuration for sweeps, offline and replays (shootout: 3 fresh-process repeats). Sweep steps are 60 s; at ≤ 3 q/s a step has ~150 requests and the p99 / achieved-rate gates are noisy (the MIG "2.6 q/s" reading).
- The load generator runs on the same host over loopback; the serving stack is a ~120-line asyncio dynamic batcher, not Triton (same algorithm). Client-observed latency includes HTTP and JSON overhead.
- The H100 was measured inside a Kubernetes pod with an Istio sidecar, 16 vCPU / 8 OMP threads, on a RHEL 8 kernel, with clocks pinned at the card's 1,785 MHz maximum; the RTX PRO 6000 bare-metal at 2,600 MHz. Eager-FP16 serving on the H100 may be CPU- or launch-bound (see §6.3) — the H100 eager rows are a lower bound on what the card can do.
- The RTX PRO 6000 sweeps have no 50 ms row and no knee refinement (sweep version predates `87417db`); the multilingual eager-FP16 shootout on the RTX 2000 Ada has one repeat.
- MIG: only the 7 × 1g.12gb layout; no 2g.24gb or 3g.47gb point; no per-slice power.
- LLM baselines report latency/throughput/tokens only, closed-loop; their first-call p99 at c=1 includes engine warm-up. Energy was not sampled for the LLM servers.
- Jev latency includes an unmeasured WAN round trip from Arizona; Jev's token accounting differs from Laya's (state once per request vs once per question).
- Cost figures rest on list-price and 100 %-utilisation assumptions stated in §6.9.
- The workload is English federal-procurement text; the multilingual checkpoint was not exercised on non-English inputs beyond the parity fixtures.
- Laya's `laya-typed-decisions` checkpoint was parity-tested only.

## 11. Gaps and follow-ups (ordered by value)

1. **`torch.compile` as the serving backend** with shape buckets (pad batch to {16, 64, 128} × sequence to {128, 256, 512}, compile once per bucket) or `dynamic=True`; the static numbers predict ~1.5× over eager and it would remove TensorRT's dynamic-shape pathology. One sweep per card.
2. **RTX PRO 6000 50 ms row**: re-run the four sweeps with the current harness (≈ 10 min each).
3. **H100 eager-serving diagnosis**: `LAYA_SERVER_DEBUG=1` to split build / forward / post per batch; try `--max-delay-ms` 5–10 and a bare-metal or non-sidecar pod.
4. **Per-length-bucket capacity**: the sweep records `seq_len`; `harness/report.py` should split p99 and cost by bucket so a short-prompt deployment can be sized.
5. **MIG 2g.24gb and 3g.47gb** points, and a short-input MIG sweep to show where slicing does pay.
6. **Jev WAN baseline**: TCP connect time from the aiohttp trace to separate network from service latency.
7. **Server correctness check**: assert a batch of N requests equals upstream `Agent.predict` one by one (planned, not automated).
8. **Cost constants**: confirm card prices and add a utilisation parameter to the notebook.
9. `notebooks/report.ipynb` executed on a clean checkout from `results/` only; README results section.

## 12. Reproduction and file map

```
./setup_env.sh .venv python3.12                 # torch 2.14 cu130, laya @ pinned commit, ORT 1.30 (CUDA-13 feed), TensorRT 10.16
hf download convaiinnovations/laya --revision c5d78730f3493e4fe16d61507ef4b78eef7318cf --local-dir models/laya
hf download convaiinnovations/laya-multilingual --local-dir models/laya-multilingual
python scripts/make_manifest.py                  # SHA256s must match manifest.json
python -m harness.envinfo --box mybox
python -m harness.parity  --box mybox --models laya laya-multilingual --backends eager-fp32 eager-fp16 ort-cuda-fp16 ort-trt-fp16 compile-fp16
python -m harness.shootout --box mybox --models laya laya-multilingual --backends eager-fp16 compile-fp16 ort-cuda-fp16 ort-trt-fp16 eager-fp32
python -m harness.sweep   --box mybox --model laya --backend ort-trt-fp16 --max-batch 128 --start-qps 50 --factor 1.4
python -m harness.offline --box mybox --model laya --backend eager-fp16 --start 2048
python -m harness.server  --model laya --backend eager-fp16 --max-batch 128 &   # then:
python -m harness.loadgen --curve fixtures/workload/diurnal.csv --total-decisions 10000000 --compress 24 --out results/mybox/replay/day.jsonl
```

Per-box chains that produced the committed results: `scripts/phase_a_laptop.sh`, `scripts/phase_b_zbook.sh`, `scripts/phase_c_rtxpro6000.sh`, `scripts/phase_d_h100_pod.sh` (inside the pod) driven by `scripts/h100_window.sh` (on the node), with the H100 checklist in `scripts/h100_window.md` and the Kubernetes manifests in `k8s/`.

```
results/
  laptop-rtx2000ada/   env.json parity.json shootout/ server/ offline/ phase_a.log
  zbook-rtxpro5000/    env.json parity.json shootout/ server/ offline/ replay/laya.eager-fp16.2M-day.jsonl phase_b.log
  rtxpro6000-ws/   env.json parity.json shootout/ server/ offline/ replay/laya.eager-fp16.10M-day.jsonl baselines/ phase_c.log
  h100nvl/             env.json parity.json (whole GPU) · env.mig-1g.12gb.json parity.mig-1g.12gb.json (slice)
                       shootout/ server/ (whole + *.mig-1g.12gb-solo + *.mig-1g.12gb-slice0..6) offline/ replay/ (7 slices) baselines/ WINDOW.md
  jev/                 probe.json probe-capped10rps.json probe-ratelimit-25rps.json c1.jsonl c8.jsonl ratelimit-25rps.c32.jsonl
  samples/             laya-vs-jev.jsonl
```

Row schemas: shootout rows `{box, gpu, model, backend, batch, length, repeat, iter, ms, n_decisions, seq_len, tokens, first_call_s, peak_vram_mb, watts_mean, sm_mhz_mean, ts}`; server/replay rows `{t_sched, t_send, t_done, latency_ms, send_lag_ms, status, served_batch, n_decisions, input_tokens}`; sweep JSON `{sustained: {50: …, 130: …}, steps: [...]}` with per-step p50/p99/achieved/served batch/power/trend/saturated.

## 13. Suggested community contributions from this work

1. **laya-mlx `BENCHMARKS.md`, new "NVIDIA CUDA" section** (PR): the parity table (§6.1 — same 63-question set, all backends 63/63, FP16 error 0.5–10e-3, BF16 up to 16e-3, TensorRT 2–19e-3 depending on engine), and the short-input latency/throughput rows in the port's own format (1 question / 64 questions, FP16, per GPU) taken from the shootout batch-1 and batch-64 short cells; link this repository for the method and raw data.
2. **Upstream `laya` README / BENCHMARKS** (PR or issue): (a) a CUDA serving section — decisions/s under p99 ≤ 50 / 130 ms per GPU (§6.3) and the recommendation "TensorRT or `torch.compile` on Hopper, eager FP16 on Blackwell, cap the batch by the SLO"; (b) note that the default `amp_dtype: bf16` gives the largest probability error of the tested dtypes on Ampere/Blackwell/Hopper (still argmax-exact) and that FP16 is tighter; (c) note `TORCH_DISABLE_NATIVE_JIT=1` for torch ≥ 2.14 if users want stock kernels; (d) the T4 rows can be placed next to these for scale.
3. **Issue on upstream: multi-state batching API.** `Agent.predict` batches one state's questions; a supported `predict_batch(states, questions)` using `build_sequence` padding would make the serving pattern here first-class.
4. **Issue/PR on upstream or a separate repo: ONNX export with dynamic axes** (this `harness/export_onnx.py`) plus the ORT/TensorRT gotchas (TF32 default, TRT 10 pin, LayerNorm FP32 fallback, provider assertion).
5. **Blog post** (outline): System 1 models in two paragraphs → why the published numbers stop at a T4 and an M3 Max and why raw latency is the wrong question for a 421M model → parity → capacity under SLO per GPU (§6.3 table + knee explanation) → the 10M-day replay → one sliced H100 vs one whole workstation card (§6.6) → what it costs vs Jev and an LLM (§6.9) → recommendations (§7) → limitations (§10).
6. **Data artefacts** worth sharing on their own: the 1,000-notice SAM.gov workload with length buckets (`fixtures/workload/`), the diurnal curve, and the harness (Apache-2.0).
