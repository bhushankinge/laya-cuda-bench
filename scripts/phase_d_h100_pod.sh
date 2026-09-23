#!/usr/bin/env bash
# Runs INSIDE the H100 bench pod (pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime); scripts/h100_window.sh on the
# node creates the pod, copies the tarball + overlay in and invokes this per stage:
#   kubectl exec <pod> -- bash -c 'cd /work && bash scripts/phase_d_h100_pod.sh <stage> [args]'
# Stages: setup | parity | whole | slice <label> | sweep-only <label> | replay <label> <total-decisions> <compress>
# Results land in /work/results/<box>/ ; the node pulls them with `h100_window.sh pull <pod>`.
set -u
cd /work
export HOME=/tmp HF_HOME=/tmp/hf OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PIP_BREAK_SYSTEM_PACKAGES=1   # the image's python is Debian-managed (PEP 668); we install into the throwaway container
export PIP_CACHE_DIR=/work/pipcache  # copied out after the first pod's setup and back into the slice pods: no 7x downloads
export LAYA_TRT_MAX_BATCH=${LAYA_TRT_MAX_BATCH:-256}
BOX=${BOX:-h100nvl}
PY=python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
stage=${1:-setup}; label=${2:-}

sweeps() {  # one MIG slice (or one of seven): both checkpoints, eager FP16, small batches
  for m in laya laya-multilingual; do
    log "sweep $m eager-fp16 [$label]"
    $PY -m harness.sweep --box $BOX --model $m --backend eager-fp16 --max-batch 64 --start-qps 10 --factor 1.4 --warmup 20 --duration 60 --label "$label" 2>&1 | grep -E "target|SLO|wrote|rror"
  done
}

case "$stage" in
setup)
  log "pip install (torch already in image)"
  pip install -q -r <(grep -vE "^torch==|^laya @" requirements.txt) && pip install -q ./third_party/laya
  pip install -q flatbuffers coloredlogs sympy protobuf packaging
  pip install -q --no-deps "onnxruntime-gpu==1.30.0" --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
  pip install -q "tensorrt-cu13==10.16.1.11"
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
  export LAYA_TRT_MAX_BATCH=64
  $PY -m harness.parity --box $BOX --models laya --backends eager-fp16 --repeats 20 2>&1 | grep -E "argmax|wrote|rror"
  sweeps
  ;;
sweep-only) sweeps ;;
replay)
  # large-org day on this slice: 1/7 of 10M/day = 1.43M decisions, 24 h -> 30 min (compress 48)
  total=${3:-1430000}; compress=${4:-48}
  mkdir -p results/$BOX/replay
  $PY -m harness.server --model laya --backend eager-fp16 --max-batch 64 --max-delay-ms 2 --port 8080 > results/$BOX/replay-server.$label.log 2>&1 &
  SRV=$!
  until $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" 2>/dev/null; do sleep 5; done  # image has no curl
  $PY -m harness.loadgen --url http://127.0.0.1:8080/predict --model laya --curve fixtures/workload/diurnal.csv --total-decisions "$total" --compress "$compress" \
    --out results/$BOX/replay/laya.eager-fp16.${label}.jsonl 2>&1 | tail -15
  kill $SRV; wait $SRV 2>/dev/null
  ;;
*) echo "unknown stage $stage"; exit 2;;
esac
log "stage $stage done"
