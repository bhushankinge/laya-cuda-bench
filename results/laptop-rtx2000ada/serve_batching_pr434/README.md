# laya-serve request batching (upstream PR #434) before and after, on an 8 GB laptop card

Upstream PR [NandhaKishorM/laya#434](https://github.com/NandhaKishorM/laya/pull/434) adds an opt-in dynamic batcher to
`laya-serve`: concurrent `POST /v1/systemone` requests arriving within `LAYA_BATCH_WINDOW_MS` are coalesced, up to
`LAYA_BATCH_MAX` requests, into one `Router.predict_batch` call. These runs measure it against `main` on the smallest card
in this study, with the same open-loop load generator and natural length mix as the sweeps in `../server/`.

Host: NVIDIA RTX 2000 Ada Generation Laptop GPU 8 GB (power-capped ~45 W), driver 590.48.01, torch 2.14.0+cu130,
transformers 5.17.0, `TORCH_DISABLE_NATIVE_JIT=1`, `english` checkpoint at the pinned revision, stock forward (the
checkpoint's bf16 autocast). `main` = `970dc8c`, PR head = `b78c581`. Both served through
`laya.serve.create_app(router=Router(models={"english": <local dir>}, device="cuda"))` (`scripts/serve_batching/serve_local.py`),
so the only difference between columns is the checkout on `PYTHONPATH` and the two env vars.

Load: `harness.loadgen` against `/v1/systemone`, Poisson arrivals at the offered rate, one state + the 3-question bundle
(`fixtures/workload/questions.json`) per request, states drawn from the 749 notices of at most 472 `laya` tokens, 40 s per
rate after an 8 s warm-up, one 10 s warm run per server first. Per-request rows are in `<config>/qps<rate>.jsonl`, the
summaries in `<config>/qps<rate>.summary.json` (`scripts/serve_batching/run_config.sh`; table by
`scripts/serve_batching/summarize.py`).

| offered q/s | main-970dc8c | pr434-defaults | pr434-w10-b16 | pr434-w10-b4 |
|---|---|---|---|---|
| 3 | 2.7 q/s, p50 50, p99 120 ms | 2.7 q/s, p50 50, p99 120 ms | 2.7 q/s, p50 63, p99 158 ms | 2.7 q/s, p50 63, p99 160 ms |
| 6 | 5.9 q/s, p50 49, p99 130 ms | 5.9 q/s, p50 49, p99 126 ms | 5.9 q/s, p50 67, p99 196 ms | 5.9 q/s, p50 65, p99 201 ms |
| 10 | 9.8 q/s, p50 58, p99 233 ms | 9.8 q/s, p50 60, p99 237 ms | 9.8 q/s, p50 84, p99 690 ms | 9.8 q/s, p50 84, p99 604 ms |
| 15 | 15.4 q/s, p50 91, p99 337 ms | 15.4 q/s, p50 90, p99 334 ms | 12.0 q/s, p50 5707, p99 12515 ms, trend 4.4x | 14.4 q/s, p50 725, p99 3370 ms, trend 4.6x |
| 22 | 20.2 q/s, p50 2005, p99 5273 ms, trend 5.1x | 20.1 q/s, p50 2038, p99 5392 ms, trend 5.1x | 10.3 q/s, p50 17965, p99 30269 ms, 300 err, trend 2.7x | 13.6 q/s, p50 13503, p99 26399 ms, trend 3.8x |

`pr434-w10-b16` = `LAYA_BATCH_WINDOW_MS=10 LAYA_BATCH_MAX=16` (the README's example values); `pr434-w10-b4` = window 10 ms,
max 4. "err" are client-side 30 s timeouts (status 599 in the JSONL); every response the server sent was 200. "trend" is
the latency ratio between the last and first third of the step (queue growing within the step).
Decisions/s: `main` 8.0 / 17.6 / 29.4 / 46.2 / 60.6; max 16: 8.0 / 17.6 / 29.3 / 35.9 / 31.0; max 4: 8.0 / 17.6 / 29.3 / 43.1 / 40.7.

Reading: the PR's defaults are identical to `main` at every rate. On this card batching loses at every rate and every batch
size tried: ~13 ms of p50 at low load (the window plus padding a mixed-length batch to its longest state), and a collapse at
15 q/s where `main` still holds, because one padded batch of long notices holds the single worker long enough for the queue
to run away. The static shootout (`../shootout/`) says why: on this power-capped card a 16-row batch of 400+-token states is
32 decisions/s against 27 at batch 1, so there is almost nothing to pay the padding with; on an H100 NVL the same cells are
280 vs 88 (long) and 835 vs 93 (short). The right `LAYA_BATCH_MAX` is therefore card- and SLO-dependent (REPORT.md §7 item 3).

## Rows, not requests (`scripts/serve_batching/rowcap_probe.py`, `probe.log`)

`LAYA_BATCH_MAX` bounds requests, but `Agent.predict_batch` collates requests × questions rows, and rows set forward time and
memory. 16 concurrent long requests with the same question set (so they coalesce), noul questions, one warm request of the
same shape first; `main` serves them as 16 sequential forwards, the PR at window 10 ms / max 16 as one forward:

| questions per request | rows in the forward | main (16 sequential forwards) | PR, window 10 ms / max 16 (one forward) |
|---|---|---|---|
| 3 | 48 | 1,183 ms wall, 62–80 ms each | 1,371 ms wall, 1,364 ms each |
| 8 | 128 | 3,347 ms wall, 193–230 ms each | 3,853 ms wall, one forward of 3,846 ms (fits; the next single requests were back at 67–70 ms on the GPU) |
| 16 | 256 | 6,527 ms wall, 384–437 ms each | CUDA OOM (four failed allocations of 0.5–1.3 GB), then one forward of 319,560 ms on the CPU via the Agent's OOM fallback, process at 9.5 GB resident |
| 64 (`MAX_QUESTIONS`) | 1,024 | 25,817 ms wall, 1,517–1,728 ms each | not attempted through the batcher after the row above |

Another card-independent point: a client that stays inside the per-request `MAX_QUESTIONS = 64` can hand the batcher a
1,024-row forward at `LAYA_BATCH_MAX=16`. The 256-row batch is not caught by the PR's per-request fallback: `Agent._infer` catches the CUDA OOM first, moves the
model to the CPU and re-runs the same batch there (`probe-w10-b16.server.log` has the four failed allocations at 14:19:23;
the `print` that announces the fallback is block-buffered under redirection and never reached the log). At `970dc8c` that
demotion is permanent for the process (upstream #344; fix #349 open). The first attempt at this shape was killed by the host
kernel's OOM killer at 10 GB resident; the rerun completed at 9.5 GB and was then stopped by a memory guard, so the 1,024-row
shape was not attempted through the batcher. The demotion itself was not measured: the 128-row batch fit and single requests after it ran on the GPU again
(67–70 ms); after the 256-row batch the process was stopped before a single request could be sent.
