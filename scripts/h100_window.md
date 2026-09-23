# H100 evening window — timed checklist (the GPU cluster <gpu-node>, namespace `<your-namespace>`)

All kubectl on the node: `ssh -i ~/.ssh/cluster_worker pcadmin@<internal-ip>`, then
`export KUBECONFIG=/etc/kubernetes/admin.conf; K="sudo --preserve-env=KUBECONFIG kubectl"`.
`nvidia-smi` on the node = `sudo chroot /run/nvidia/driver nvidia-smi`. Every prod-touching step is
confirmed live with the user before running. Hard rule: restore starts no later than T+5:30.

## Daytime prep (no downtime)
- [ ] `git status` clean on the dev laptop; results from laptop / ZBook / workstation committed (pipeline proven).
- [ ] Snapshot to `~/incident-logs/h100-window-<date>/`: `$K get node <gpu-node> -o yaml`,
      `$K get clusterpolicies.nvidia.com cluster-policy -o yaml`, `$K -n gpu-operator get cm custom-mig-config -o yaml`,
      `$K get gpumigconfigs -A -o yaml`, `$K -n <your-namespace> get resourcequota,isvc,revisions -o yaml`.
- [ ] Merge `k8s/mig-configs.yaml` entries into `custom-mig-config` (additive; inert until the node label changes).
- [x] (2026-09-22, setup stage 368 s, `PIP_BREAK_SYSTEM_PACKAGES=1` needed) Dry run `k8s/bench-pod.yaml` with the GPU limit removed: image pull, `kubectl cp` of `harness/ fixtures/ models/laya models/laya-multilingual requirements.txt setup_env.sh`,
      `pip install` (torch is already in the image: install `-r requirements.txt` minus torch, then ORT cu13 feed + tensorrt-cu13), `python -m harness.workload`. Delete the pod.
- [ ] Colleague informed: `qwen3-8b` restarts at the MIG transition (~T+2:50) and at restore (~T+4:40).
- [ ] Tarball ready: `tar czf /tmp/laya-bench.tgz harness fixtures models/laya models/laya-multilingual requirements.txt setup_env.sh` (weights 1.5 GB) — or `hf download` inside the pod.

## Evening (T = 18:00)
| T+ | Step | Verify before moving on |
|---|---|---|
| 0:00 | Qwen3.5-35B-A3B baseline via live VIP: `python -m harness.llm_baseline --box h100nvl --label qwen3.5-35b-a3b-fp8 --base-url https://<cluster-llm-endpoint>/v1 --llm-model <served name> --api-key-env the GPU cluster_TOKEN --no-think` (token = `<ingress-token>` from secret `access-token`; add `/etc/hosts` or `--resolve`) | 3 levels written to `results/h100nvl/baselines/` |
| 0:20 | On workstation: `<llm-failover-script>` → standby; smoke `curl workstation:8000/v1/chat/completions` | application answers from workstation |
| 0:30 | `$K apply -f k8s/kyverno-hold.yaml`; `$K -n <your-namespace> delete pod -l serving.kserve.io/inferenceservice=qwen3-5-35b-a3b-fp8`; delete superseded revisions | `nvidia-smi`: GPU 0 idle, 0 MiB |
| 0:35 | Pin clocks: `sudo chroot /run/nvidia/driver nvidia-smi -i 0 -lgc 1785,1785` | `clocks.sm` reads 1785 |
| 0:40 | `$K apply -f k8s/bench-pod.yaml` (nvidia.com/gpu: 1); `kubectl cp` tarball (`~/laya-bench-prep/laya-bench.tgz`, 13 s) and extract; then `kubectl cp ~/laya-bench-prep/harness-overlay.tgz <pod>:/work/ -c bench` and `tar xzf harness-overlay.tgz` in /work (overlays the fixed harness/, scripts/, README over the big tarball, which predates the replay-rate, sweep and PEP668 fixes); `bash scripts/phase_d_h100_pod.sh setup` (368 s in the dry run) | env.json shows H100 NVL, driver 580.95; `trt 10.16.1.11` printed |
| 0:45 | **Parity gate**: `python -m harness.parity --box h100nvl --backends eager-fp16 eager-fp32 ort-cuda-fp16 ort-trt-fp16 compile-fp16 --models laya laya-multilingual` | 63/63 everywhere → GO; else stop and restore |
| 0:55 | Shootout: `python -m harness.shootout --box h100nvl --models laya laya-multilingual` (~40 min) | JSONL under `results/h100nvl/shootout/` |
| 1:35 | Server sweeps: `python -m harness.sweep --box h100nvl --model laya --backend <winner> --start-qps 50 --factor 1.4`; same for `laya-multilingual` (~15 min each) | `*.sweep.json` with sustained rows for 50/130 ms |
| 2:05 | Offline: `python -m harness.offline --box h100nvl --model laya --backend <winner> --start 2048`; same for multilingual | offline JSON |
| 2:15 | `kubectl cp` results out → `results/h100nvl/`; commit on the laptop. Delete bench pod. | git shows the files |
| 2:20 | `$K apply -f k8s/vllm-4b-pod.yaml`; wait Ready (download ~8 GB); port-forward 8010; `llm_baseline --label qwen3.5-4b`; delete pod | baselines JSON |
| 2:50 | MIG: `$K patch clusterpolicies.nvidia.com cluster-policy --type merge -p '{"spec":{"mig":{"strategy":"mixed"}}}'`; `$K -n gpu-nodeconfig scale deploy gpunodeconfig-controller-manager --replicas=0`; colleague's predictor pod deleted (agreed); `$K label node <gpu-node> nvidia.com/mig.config=mixed-bd --overwrite`; watch `$K -n gpu-operator logs ds/nvidia-mig-manager -f` | node label `mig.config.state=success`; `nvidia-smi -L` shows 5 MIG devices on GPU 0; colleague's pod back on GPU 1 |
| 3:05 | Bench pod with `nvidia.com/mig-2g.24gb: "1"`: parity (eager-fp16 + winner), `sweep --label mig-2g.24gb --start-qps 20`; then same pod spec with `mig-1g.12gb`, `--label mig-1g.12gb --start-qps 10` | two sweep JSONs |
| 3:45 | `$K label node ... nvidia.com/mig.config=all-1g-bd --overwrite` (GPU 0 idle first) → 7 pods from `k8s/bench-pod.yaml` named `laya-bench-0..6`, each `mig-1g.12gb: 1`; start all 7 servers, one loadgen per pod, shared start time; `sweep --no-server` per pod at the same target ladder; aggregate = sum of per-pod sustained decisions/s | 7 sweep JSONs with `--label mig-1g.12gb-slice<i>` |
| 4:15 | 30-minute replay on the 7 slices: `loadgen --curve fixtures/workload/diurnal.csv --total-decisions 1.43e6 --compress 48` per pod (1/7 of 10M/day, 24 h → 30 min) | replay JSONL per pod |
| 4:40 | **Restore**: delete bench pods; `label nvidia.com/mig.config=whole` → `state=success`; strategy back to `single`; restore `custom-mig-config` from snapshot; controller `--replicas=1`; `nvidia-smi -i 0 -rgc`; `$K delete clusterpolicy laya-bench-hold-gpu`; the managed inference service deployment returns (~25 min re-download) | predictor pod Ready; `nvidia-smi`: GPU 0 = Qwen3.5-35B, GPU 1 = qwen3-8b |
| 5:10 | Smoke via VIP (authenticated chat completion); workstation `<llm-failover-script>` back to cluster; application smoke check | `LLM_BACKEND=cluster`, answer returned |
| 5:20 | `kubectl cp` all results out; commit. Write the window log (what ran, what was skipped) into `results/h100nvl/WINDOW.md`. | |

## Abort rules
- Parity not 63/63 on the H100 → do not benchmark; restore immediately, debug offline.
- Any step blocked > 20 min past its slot → skip forward; MIG is skipped entirely if T+3:30 is reached without the `mixed-bd` transition done.
- MIG transition `ERROR_IN_USE` → check `nvidia-smi` for leftover processes, delete them via their pods, delete the mig-manager pod to re-trigger; never touch the colleague's namespace beyond the agreed pod deletion.
