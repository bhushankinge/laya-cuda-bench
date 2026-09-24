# H100 NVL window log — 2026-09-23 (Phoenix times)

GPU 0 of a 2× H100 NVL 94 GB node (managed Kubernetes cluster, driver 580.95, CUDA 13), used from a plain Kubernetes pod with
`pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime`. Clocks pinned at 1785 MHz (the NVL's max SM clock) for the
whole-GPU segment. Box label `h100nvl`. Driver script: `scripts/h100_window.sh`, pod stages: `scripts/phase_d_h100_pod.sh`.

| time | step | outcome |
|---|---|---|
| 11:33 | Qwen3.5-35B-A3B-FP8 baseline through the platform ingress | 15 decisions/s flat at c=16 and c=64 (requests served one at a time by the serving path); kept as `qwen3.5-35b-a3b-fp8.json` |
| 11:45 | same model, port-forward straight to the vLLM container | 200 decisions/s at c=64, p99 1.06 s: `qwen3.5-35b-a3b-fp8-direct.json` |
| 11:50 | LLM traffic failed over to the standby server; GPU-0 predictor held down; clocks pinned | GPU 0 at 0 MiB |
| 11:55–12:13 | whole-GPU pod, pip install (17 min) | env.json |
| 12:13–13:05 | parity gate | 63/63 on eager FP16/FP32, ORT CUDA, ORT TensorRT (256 profile), torch.compile — both checkpoints; compile needed a user-space g++ (`gxx` stage) because the image has gcc but no g++ |
| 13:09–14:29 | shootout, 5 backends × 2 checkpoints × 4 batches × 2 lengths × 3 repeats | `shootout/` |
| 14:29–15:18 | sweeps eager FP16 + ORT TensorRT FP16 for both checkpoints (max-batch 128, from 50 q/s) | `server/*.sweep.json` |
| 15:18–15:22 | offline ceilings, batch 2048 | `offline/` |
| 15:30 | Qwen3.5-4B (vLLM 0.24 pod, direct to container) | 200 decisions/s at c=64, p99 0.99 s |
| 15:39 | MIG: GPU 0 → 7 × 1g.12gb (`all-1g-bd`), GPU 1 untouched | label `success` in < 1 min; slices advertised ~1 min later |
| 15:40–15:44 | 7 slice pods, installed from the first pod's pip cache | no downloads |
| 15:45–16:02 | slice 0 alone: parity (eager FP16) + sweeps both checkpoints, max-batch 64, from 10 q/s | `*.mig-1g.12gb-solo.sweep.json` |
| 16:02–16:19 | all 7 slices sweeping concurrently | `*.mig-1g.12gb-slice<i>.sweep.json` |
| 16:19–16:49 | 30-min replay per slice: 1/7 of a 10M-decisions/day diurnal curve, 48× compressed | `replay/*.mig-1g.12gb-slice<i>.jsonl` |
| 16:50 | restore: pods deleted, MIG `whole`, strategy `single`, original MIG config, controller back, clocks reset, hold removed | predictor re-download started 16:50 |

## Skipped
- The `mixed-bd` layout (one 2g.24gb + four 1g.12gb slices): optional in the plan, dropped for time.
- torch.compile was not swept as a serving backend (recompiles per batch shape); it is the static winner and a follow-up item.
- No power readings inside MIG pods: NVML reports 0 W for a MIG device; power is per physical card only.

## Headline numbers
- Whole H100, `laya`, natural length mix: **ORT TensorRT FP16 sustains 105 decisions/s under p99 ≤ 50 ms (9.0M/day, 193 W) and 175 decisions/s under p99 ≤ 130 ms (15.1M/day, 236 W)**; eager FP16 39 / 93 decisions/s.
- A single 1g.12gb slice does not meet either SLO at any rate on this mix: at 2.6 q/s p99 is 126 ms (`laya`) / 130 ms (multilingual) with p50 47 / 30 ms; a single long notice takes longer than the loose SLO on one seventh of the card. All seven concurrent slices reproduce the solo numbers to within 1 ms (MIG isolation holds).
- Seven slices carried the 10M/day curve with zero errors: pooled p50 51 ms, p90 127 ms, **p99 285 ms**, worst 150-s window p99 367 ms; 117 decisions/s over the 30 minutes.
- LLMs on the same card: Qwen3.5-35B-A3B-FP8 and Qwen3.5-4B both ~200 decisions/s at c=64 with p99 ~1 s.
