"""laya-cuda-bench harness: one predict() path, swappable forward backends."""
import os
from pathlib import Path

# torch 2.14 routes some eager ops (e.g. the rotary-embedding bmm) to Triton kernels via torch._native.
# "eager" rows here mean stock aten kernels on every box, so that JIT is off; torch.compile is its own row.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures"
CHECKPOINTS = ("laya", "laya-multilingual", "laya-typed-decisions")
