#!/usr/bin/env bash
# Laptop (RTX 2000 Ada, 8 GB) Phase A chain, run detached: sweeps -> offline -> reduced shootout -> parity rebuild.
# Usage: setsid nohup scripts/phase_a_laptop.sh > results/laptop-rtx2000ada/phase_a.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export LAYA_TRT_MAX_BATCH=32
BOX=laptop-rtx2000ada
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

for m in laya laya-multilingual; do
  log "sweep $m"
  $PY -m harness.sweep --box $BOX --model $m --backend eager-fp16 --start-qps 1 --factor 1.5 --warmup 15 --duration 45 2>&1 | grep -E "target|SLO|wrote|rror"
  log "offline $m"
  $PY -m harness.offline --box $BOX --model $m --backend eager-fp16 --start 256 2>&1 | grep -E "batch|decisions|rror"
done

log "shootout (reduced: batches 1 16 64, 1 repeat per backend)"
$PY -m harness.shootout --box $BOX --models laya laya-multilingual --backends eager-fp16 compile-fp16 ort-cuda-fp16 ort-trt-fp16 eager-fp32 \
  --batches 1 16 64 --lengths short long --repeats 1 2>&1 | grep -E "^(==>|skip|laya|child)"
log "shootout (eager-fp16 laya, 3 repeats)"
$PY -m harness.shootout --box $BOX --models laya --backends eager-fp16 --batches 1 16 64 --lengths short long --repeats 3 2>&1 | grep -E "^(==>|skip|laya|child)"

log "parity rebuild"
rm -rf models/laya/trt_cache
$PY -m harness.parity --box $BOX --backends eager-fp32 eager-fp16 eager-bf16 ort-cuda-fp32 ort-cuda-fp16 compile-fp16 --repeats 100 2>&1 | grep -E "argmax|wrote|rror"
$PY -m harness.parity --box $BOX --models laya --backends ort-trt-fp16 --repeats 100 2>&1 | grep -E "argmax|wrote|rror"
log "done"
