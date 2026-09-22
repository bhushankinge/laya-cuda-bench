#!/usr/bin/env bash
# workstation (RTX PRO 6000 Blackwell 96 GB) exclusive-window chain. Run ON workstation AFTER `<exclusive-window-script> stop`
# has verified 0 MiB used (and LLM_BACKEND=cluster):
#   cd ~/laya-bench && setsid nohup scripts/phase_c_rtxpro6000.sh > results/rtxpro6000-ws/phase_c.log 2>&1 &
# Then `<exclusive-window-script> start` once this prints "done". The Qwen3.5-35B-A3B baseline is taken separately
# while the standby LLM is still up (harness.llm_baseline against :8000, done from the dev laptop).
set -u
cd "$(dirname "$0")/.."
PY=~/venvs/laya-bench/bin/python
export LAYA_TRT_MAX_BATCH=256
BOX=rtxpro6000-ws
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 1024 ] || { echo "GPU not exclusive (${used} MiB used); run <exclusive-window-script> stop first"; exit 1; }

log "clock pin attempt (recorded either way)"
sudo nvidia-smi -lgc 2610,2610 2>&1 | tail -1 || true
log "envinfo"; $PY -m harness.envinfo --box $BOX | tail -1
log "TensorRT parity at the full 256 profile (engine used for timing)"
rm -rf models/*/trt_cache
$PY -m harness.parity --box $BOX --models laya laya-multilingual --backends ort-trt-fp16 --repeats 100 2>&1 | grep -E "argmax|wrote|rror"

log "shootout (full)"
$PY -m harness.shootout --box $BOX --models laya laya-multilingual --backends eager-fp16 compile-fp16 ort-cuda-fp16 ort-trt-fp16 eager-fp32 \
  --batches 1 16 64 256 --lengths short long --repeats 3 2>&1 | grep -E "^(==>|skip|laya|child)"

for m in laya laya-multilingual; do
  for b in eager-fp16 ort-trt-fp16; do
    log "sweep $m $b"
    $PY -m harness.sweep --box $BOX --model $m --backend $b --max-batch 128 --start-qps 50 --factor 1.4 --warmup 20 --duration 60 2>&1 | grep -E "target|SLO|wrote|rror"
  done
  log "offline $m"
  $PY -m harness.offline --box $BOX --model $m --backend eager-fp16 --start 2048 2>&1 | grep -E "batch|decisions|rror"
done

log "large-org day replay (10M decisions/day, 24x compressed, 1 h)"
$PY -m harness.server --model laya --backend eager-fp16 --max-batch 128 --max-delay-ms 2 --port 8080 > results/$BOX/replay-server.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8080/healthz >/dev/null; do sleep 5; done
mkdir -p results/$BOX/replay
$PY -m harness.loadgen --url http://127.0.0.1:8080/predict --model laya --curve fixtures/workload/diurnal.csv --total-decisions 10000000 --compress 24 \
  --out results/$BOX/replay/laya.eager-fp16.10M-day.jsonl 2>&1 | tail -15
kill $SRV; wait $SRV 2>/dev/null

log "Qwen3.5-4B temporary vLLM on :8010 (vLLM 0.17.1 venv)"
HF_HOME=~/hf-cache ~/venvs/vllm/bin/vllm serve Qwen/Qwen3.5-4B --port 8010 --max-model-len 8192 --gpu-memory-utilization 0.6 \
  --served-model-name Qwen/Qwen3.5-4B > results/$BOX/qwen4b-vllm.log 2>&1 &
VL=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8010/v1/models >/dev/null && break; sleep 10; done
$PY -m harness.llm_baseline --box $BOX --label qwen3.5-4b --base-url http://127.0.0.1:8010/v1 --llm-model Qwen/Qwen3.5-4B \
  --concurrency 1 16 64 --requests 60 120 240 --no-think 2>&1 | grep -E "c=|wrote|rror"
kill $VL; wait $VL 2>/dev/null
sudo nvidia-smi -rgc 2>&1 | tail -1 || true
log "done"
