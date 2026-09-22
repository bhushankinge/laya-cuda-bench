"""laya-cuda-bench harness: one predict() path, swappable forward backends."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures"
CHECKPOINTS = ("laya", "laya-multilingual", "laya-typed-decisions")
