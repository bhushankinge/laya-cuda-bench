#!/usr/bin/env bash
# Second ZBook (RTX PRO 5000 Blackwell Laptop 24 GB), exclusive GPU. Run detached on the ZBook:
#   cd ~/laya-bench && setsid nohup scripts/phase_b_zbook.sh > results/zbook-rtxpro5000/phase_b.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CPATH=$HOME/pyinc/usr/include/python3.12:$HOME/pyinc/usr/include   # Triton/Inductor need Python.h (no python3-dev here)
export LAYA_TRT_MAX_BATCH=256
BOX=zbook-rtxpro5000
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "envinfo"; $PY -m harness.envinfo --box $BOX | tail -1
log "shootout (full: batches 1 16 64 256, 3 repeats, 5 backends, 2 checkpoints)"
rm -rf models/*/trt_cache
$PY -m harness.shootout --box $BOX --models laya laya-multilingual --backends eager-fp16 compile-fp16 ort-cuda-fp16 ort-trt-fp16 eager-fp32 \
  --batches 1 16 64 256 --lengths short long --repeats 3 2>&1 | grep -E "^(==>|skip|laya|child)"

for m in laya laya-multilingual; do
  for b in eager-fp16 ort-trt-fp16; do
    log "sweep $m $b"
    $PY -m harness.sweep --box $BOX --model $m --backend $b --start-qps 20 --factor 1.5 --warmup 20 --duration 60 2>&1 | grep -E "target|SLO|wrote|rror"
  done
  log "offline $m"
  $PY -m harness.offline --box $BOX --model $m --backend eager-fp16 --start 1024 2>&1 | grep -E "batch|decisions|rror"
done

log "large-org day replay: 10M decisions/day compressed 24x into 1 h, laya eager-fp16"
$PY -m harness.server --model laya --backend eager-fp16 --max-batch 64 --max-delay-ms 2 --port 8080 > results/$BOX/replay-server.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8080/healthz >/dev/null; do sleep 5; done
mkdir -p results/$BOX/replay
$PY -m harness.loadgen --url http://127.0.0.1:8080/predict --model laya --curve fixtures/workload/diurnal.csv --total-decisions 10000000 --compress 24 \
  --out results/$BOX/replay/laya.eager-fp16.10M-day.jsonl 2>&1 | tail -15
kill $SRV; wait $SRV 2>/dev/null
log "done"
