# Stock forward: bf16 autocast (the CUDA default) vs fp16 autocast vs fp32

Upstream's `Agent` autocasts in the checkpoint's `amp_dtype` on any CUDA GPU with compute capability >= 8, and all three
shipped checkpoints set `bf16`. These runs answer upstream's own fixed parity set (`benchmarks/parity_fast.py`: 60 states,
up to 8 questions each, 288 questions per checkpoint) with the stock forward under bf16 autocast, under fp16 autocast, and
in fp32 as the reference. Same tokens, same model call, same batching as `parity_fast.py` (one padded batch per state),
only the autocast dtype changes. Probabilities are the raw softmax of the logits, before temperature scaling, exactly as
`parity_fast.py` reports them.

Host: NVIDIA RTX 2000 Ada Generation Laptop GPU (cc 8.9), driver 590.48.01, torch 2.14.0+cu130, transformers 5.17.0,
laya at `970dc8c`, `TORCH_DISABLE_NATIVE_JIT=1`. Run: `LAYA_SRC=<laya checkout> python scripts/dtype_parity.py <checkpoint dir>`.

| checkpoint | type | n | max \|p_bf16 - p_fp32\| | max \|p_fp16 - p_fp32\| | argmax bf16 = fp32 | fp16 = fp32 |
|---|---|---|---|---|---|---|
| laya | choice | 48 | 0.0185 | 0.0076 | 48/48 | 48/48 |
| laya | noul | 180 | 0.0731 | 0.0185 | 179/180 | 180/180 |
| laya | score | 60 | 0.0116 | 0.0052 | 60/60 | 60/60 |
| laya-multilingual | choice | 48 | 0.0391 | 0.0019 | 48/48 | 48/48 |
| laya-multilingual | noul | 180 | 0.0709 | 0.0068 | 180/180 | 180/180 |
| laya-multilingual | score | 60 | 0.0079 | 0.0021 | 60/60 | 60/60 |
| laya-typed-decisions | choice | 48 | 0.0124 | 0.0032 | 46/48 | 48/48 |
| laya-typed-decisions | noul | 180 | 0.0419 | 0.0045 | 180/180 | 180/180 |
| laya-typed-decisions | score | 60 | 0.0128 | 0.0013 | 60/60 | 60/60 |

Median per-question error: bf16 0.0013 / 0.0005 / 0.0018 vs fp16 0.0001 / 0.0001 / 0.0002 (laya / multilingual / typed-decisions).
End-to-end `predict()` latency, one short state x 3 triage questions, 100 iterations after warm-up (p50 / p95 ms):
laya bf16 31.3 / 34.5 vs fp16 31.2 / 33.6; multilingual 14.1 / 15.8 vs 13.5 / 14.7; typed-decisions 31.6 / 33.6 vs 32.2 / 34.2.

The three argmax flips (every per-option probability is in the JSON files):

| checkpoint | state / question | fp32 | bf16 | fp16 |
|---|---|---|---|---|
| laya | moderation/1 spam (noul) | 0.562 / 0.438 | 0.489 / 0.511 | 0.562 / 0.438 |
| laya-typed-decisions | guard/5 topic (choice) | 0.2375 vs 0.2365 | 0.2368 vs 0.2382 | 0.2374 vs 0.2365 |
| laya-typed-decisions | email/6 category (choice) | 0.3092 vs 0.3081 | 0.3066 vs 0.3096 | 0.3085 vs 0.3079 |

This is the same picture as the 63-question parity gate in `../parity.json` and REPORT.md section 6.1 (bf16 0.004-0.016 vs
fp16 0.0005-0.007 across four GPUs, 63/63 argmax on that smaller set); the larger upstream set is where the flips appear.
