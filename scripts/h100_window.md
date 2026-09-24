# H100 evening window — timed checklist (Kubernetes GPU node; set NS / COLLEAGUE_NS / NODE at the top of `h100_window.sh`)

Driver: `scripts/h100_window.sh` runs ON the GPU node from `~/laya-bench-prep/` (staged there with `k8s/`,
`laya-bench.tgz` = harness+fixtures+weights, `harness-overlay.tgz` = current harness/scripts/k8s over it).
`h100_window.sh status` at any time. Every prod-touching subcommand (`hold`, `colleague-restart`, `mig`,
`restore`, and the application's LLM failover) is confirmed live with the user before running. Hard rule: `restore`
starts no later than T+5:30. Namespace quota is 40 CPU / 100 GiB of *limits*: whole-GPU pods use 16 CPU / 64 GiB,
the seven slice pods 5 CPU / 13 GiB each (`pod ... slice`).

## Daytime prep (no downtime) — done 2026-09-23
- [x] `git status` clean; laptop / ZBook / workstation results committed.
- [x] `h100_window.sh snapshot` → `~/laya-bench-prep/snapshot-<utc>/` (`snapshot-latest` symlink; `restore` reads it).
- [x] Dry run of the pod install path (2026-09-22, setup 368 s, `PIP_BREAK_SYSTEM_PACKAGES=1` needed).
- [x] Qwen3.5-35B-A3B baseline through the live ingress from the workstation → `results/h100nvl/baselines/qwen3.5-35b-a3b-fp8.json`.
      Finding: through the platform ingress requests are served one at a time (5 req/s = 15 decisions/s flat at c=16 and
      c=64, latency = concurrency × 200 ms). That column measures the platform's serving-ingress path, not the GPU; take a second
      reading direct to the predictor container during the window (step 0:00 below).
- [ ] `h100_window.sh migconfig-add` (additive keys in `custom-mig-config`, inert until the node label changes).
- [ ] Colleague informed: `qwen3-8b` restarts at every MIG transition (all-1g at ~T+2:05, restore at ~T+3:35, plus one
      more if the optional mixed-bd layout runs).

## Evening (T = 17:00 Phoenix). Order: whole GPU → Qwen 4B → 7 × 1g.12gb (central claim) → optional 2g.24gb → restore
| T+ | Step (node unless stated) | Verify before moving on |
|---|---|---|
| 0:00 | Direct 35B reading: node `kubectl -n <ns> port-forward pod/<predictor> 8011:8080`; laptop `ssh -L 8011:127.0.0.1:8011 <node>` and `python -m harness.llm_baseline --box h100nvl --label qwen3.5-35b-a3b-fp8-direct --base-url http://127.0.0.1:8011/v1 --llm-model Qwen/Qwen3.5-35B-A3B-FP8 --no-think` | 3 levels; c=64 well above 15 decisions/s if the GPU is the limit |
| 0:10 | **Workstation**: fail the application's LLM traffic over to the standby vLLM (already warm); smoke one chat completion | application answers from the standby |
| 0:15 | **`h100_window.sh hold`**: Kyverno hold, delete our predictor pod, wait GPU 0 idle, `-lgc 1785` | `status`: GPU 0 0 MiB, clocks 1785 |
| 0:20 | `h100_window.sh pod laya-bench nvidia.com/gpu` (copy 2.7 GB, extract, setup ~6 min) → `pipcache-save laya-bench` | env.json shows H100 NVL; `trt 10.16.1.11` |
| 0:30 | **Parity gate** `run laya-bench parity` | 63/63 on all 5 backends × 2 checkpoints → GO; else `restore` |
| 0:40 | `run laya-bench whole` (shootout ~40 min, 4 sweeps ~30 min, 2 offline) | JSONL under results/h100nvl/ |
| 1:55 | `pull laya-bench`; delete the pod; `qwen4b-up` (8 GB download); port-forward 8010:8080 + laptop tunnel; `llm_baseline --label qwen3.5-4b --base-url http://127.0.0.1:8010/v1 --llm-model Qwen/Qwen3.5-4B --no-think`; delete the pod | baselines JSON |
| 2:05 | **`colleague-restart`** (agreed) then **`mig all-1g-bd`**: strategy mixed, controller to 0, label, wait `state=success` | `nvidia-smi -L`: 7 MIG devices on GPU 0; allocatable `mig-1g.12gb: 7`; colleague's pod recreating on GPU 1 |
| 2:20 | `slices7 up` (7 pods from the pip cache, parallel) → `slices7 solo` (pod 0 alone = single-slice number, ~12 min) → `slices7 sweep` (all 7 at once, ~12 min) | 1 + 7 sweep JSONs, labels `mig-1g.12gb-solo`, `mig-1g.12gb-slice<i>` |
| 2:50 | `slices7 replay` (30 min: 1/7 of a 10M/day curve per pod, compress 48) → `slices7 pull` → `slices7 down` | 7 replay JSONL |
| 3:25 | *Optional if before T+3:30*: `colleague-restart`, `mig mixed-bd`, `pod laya-bench-2g nvidia.com/mig-2g.24gb`, `run laya-bench-2g slice mig-2g.24gb`, `pull`, delete pod (~30 min) | one more sweep pair |
| 3:35 | **`restore`**: pods gone → label `whole` → strategy `single` → original config → controller 1 → `-rgc` → hold deleted → waits for our predictor (~25 min) | `status`: whole/success, predictor 4/4, colleague 4/4 |
| 4:05 | **Workstation**: ingress smoke (authenticated chat completion), fail the application's LLM traffic back to the cluster, application smoke check | answer returned |
| 4:10 | `rsync` node `~/laya-bench-prep/results/h100nvl/` → laptop `results/h100nvl/`; commit; write what ran / what was skipped into `results/h100nvl/WINDOW.md` | git shows the files |

## Abort rules
- Parity not 63/63 on the H100 → do not benchmark; `restore` immediately, debug offline.
- Any step blocked > 20 min past its slot → skip forward; MIG is skipped entirely if T+3:30 is reached without `all-1g-bd` done.
- MIG transition `ERROR_IN_USE` → `status` shows the leftover process; delete it via its pod; delete the mig-manager pod to
  re-trigger; never touch the colleague's namespace beyond the agreed pod deletion.
