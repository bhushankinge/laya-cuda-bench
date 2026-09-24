#!/bin/bash
# run_config.sh <label> <laya checkout> <rates...>   env: LAYA_BATCH_WINDOW_MS / LAYA_BATCH_MAX pass through to the server
# Run from the repo root with LAYA_MODEL_DIR set. Starts laya.serve on :8001 from <laya checkout>, drives it with harness.loadgen at each rate, writes summaries.
set -u
LABEL=$1; SRC=$2; shift 2
OUT="${OUT_DIR:-results/serve_batching}/$LABEL"; mkdir -p "$OUT"
PY="${PYTHON:-python}"
export TORCH_DISABLE_NATIVE_JIT=1 HF_HUB_OFFLINE=1

PYTHONPATH="$SRC" "$PY" scripts/serve_batching/serve_local.py 8001 > "$OUT/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -sf localhost:8001/health | grep -q english && break; sleep 2; done
curl -s localhost:8001/health; echo
# warm the forward once so the first timed request is not the CUDA warm-up
"$PY" -m harness.loadgen --url http://127.0.0.1:8001/v1/systemone --model laya --qps 2 --duration 10 --warmup 10 > /dev/null 2>&1
for QPS in "$@"; do
  "$PY" -m harness.loadgen --url http://127.0.0.1:8001/v1/systemone --model laya --qps "$QPS" --duration 40 --warmup 8 \
      --out "$OUT/qps$QPS.jsonl" | tee "$OUT/qps$QPS.summary.json"
done
kill $SRV; wait $SRV 2>/dev/null
echo "done $LABEL"
