#!/usr/bin/env bash
# Runs INSIDE the H100 bench pod (pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime) after
#   kubectl cp ~/laya-bench-prep/laya-bench.tgz <pod>:/work/laya-bench.tgz
#   kubectl exec <pod> -- bash -c 'cd /work && tar xzf laya-bench.tgz && bash scripts/phase_d_h100_pod.sh <stage> [label]'
# Stages: setup | parity | whole | slice <label> | server-only <label> (7-slice aggregate: one pod per slice)
# Results land in /work/results/<box>/ ; copy out with kubectl cp <pod>:/work/results ./results-h100
set -u
cd /work
export HOME=/tmp HF_HOME=/tmp/hf OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export LAYA_TRT_MAX_BATCH=${LAYA_TRT_MAX_BATCH:-256}
BOX=${BOX:-h100nvl}
PY=python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
stage=${1:-setup}; label=${2:-}

case "$stage" in
setup)
  log "pip install (torch already in image)"
  pip install -q --no-cache-dir -r <(grep -vE "^torch==|^laya @" requirements.txt) && pip install -q --no-cache-dir ./third_party/laya
  pip install -q --no-cache-dir flatbuffers coloredlogs sympy protobuf packaging
  pip install -q --no-cache-dir --no-deps "onnxruntime-gpu==1.30.0" --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
  pip install -q --no-cache-dir "tensorrt-cu13==10.16.1.11"
  $PY -c "import torch, onnxruntime as ort, tensorrt, laya; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('ort', ort.__version__, ort.get_available_providers()); print('trt', tensorrt.__version__)"
  $PY -m harness.envinfo --box $BOX | tail -1
  $PY -m harness.workload
  ;;
parity)
  log "parity gate (go/no-go)"
  $PY -m harness.parity --box $BOX --models laya laya-multilingual --backends eager-fp16 eager-fp32 ort-cuda-fp16 ort-trt-fp16 compile-fp16 --repeats 100 2>&1 | grep -E "argmax|wrote|rror|requested"
  ;;
whole)
  log "shootout"
  $PY -m harness.shootout --box $BOX --models laya laya-multilingual --backends eager-fp16 compile-fp16 ort-cuda-fp16 ort-trt-fp16 eager-fp32 \
    --batches 1 16 64 256 --lengths short long --repeats 3 2>&1 | grep -E "^(==>|skip|laya|child)"
  for m in laya laya-multilingual; do
    for b in eager-fp16 ort-trt-fp16; do
      log "sweep $m $b"
      $PY -m harness.sweep --box $BOX --model $m --backend $b --max-batch 128 --start-qps 50 --factor 1.4 --warmup 20 --duration 60 2>&1 | grep -E "target|SLO|wrote|rror"
    done
    log "offline $m"; $PY -m harness.offline --box $BOX --model $m --backend eager-fp16 --start 2048 2>&1 | grep -E "batch|decisions|rror"
  done
  ;;
slice)
  # one MIG slice: parity (eager-fp16 only, quick) + sweeps for both checkpoints
  export LAYA_TRT_MAX_BATCH=64
  $PY -m harness.parity --box $BOX --models laya --backends eager-fp16 --repeats 20 2>&1 | grep -E "argmax|wrote|rror"
  for m in laya laya-multilingual; do
    log "sweep $m eager-fp16 [$label]"
    $PY -m harness.sweep --box $BOX --model $m --backend eager-fp16 --max-batch 64 --start-qps 10 --factor 1.4 --warmup 20 --duration 60 --label "$label" 2>&1 | grep -E "target|SLO|wrote|rror"
  done
  ;;
server-only)
  # 7-slice aggregate: each pod serves; the load generator runs from the node against each pod IP (see h100_window.md)
  exec $PY -m harness.server --model laya --backend eager-fp16 --max-batch 64 --max-delay-ms 2 --port 8080
  ;;
*) echo "unknown stage $stage"; exit 2;;
esac
log "stage $stage done"
