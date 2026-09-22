#!/usr/bin/env bash
# Reproducible environment for laya-cuda-bench. Usage: ./setup_env.sh [venv_dir] [python]
set -euo pipefail
VENV="${1:-.venv}"
PY="${2:-python3}"
if ! "$PY" -m venv "$VENV" 2>/dev/null; then
  # Ubuntu without python3-venv (no ensurepip): create the venv bare and bootstrap pip into it.
  rm -rf "$VENV"; "$PY" -m venv --without-pip "$VENV"
  curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python" -
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install -q -r "$(dirname "$0")/requirements.txt"
# ORT CUDA 13 build lives on its own feed; the same version number on PyPI is the CUDA 12 build, so the
# feed must be the only index for that one package. Its deps come from PyPI first, then --no-deps.
"$VENV/bin/pip" install -q flatbuffers coloredlogs sympy protobuf packaging
"$VENV/bin/pip" install -q --no-deps "onnxruntime-gpu==1.30.0" \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
# ORT 1.30's TensorRT EP links libnvinfer.so.10 -> newest TensorRT 10.x with CUDA 13 wheels, not 11.x.
"$VENV/bin/pip" install -q "tensorrt-cu13==10.16.1.11"
"$VENV/bin/python" - <<'EOF'
import torch, onnxruntime as ort
print("torch", torch.__version__, "cuda", torch.version.cuda, "ok" if torch.cuda.is_available() else "NO CUDA")
print("onnxruntime", ort.__version__, ort.get_available_providers())
try:
    import tensorrt; print("tensorrt", tensorrt.__version__)
except Exception as e: print("tensorrt import failed:", e)
EOF
