"""Environment manifest for one box: python -m harness.envinfo --box laptop-rtx2000ada"""
import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

from . import RESULTS


def _pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _smi(query):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], text=True, timeout=10
        )
        return [line.strip() for line in out.strip().splitlines()]
    except Exception as e:  # nvidia-smi absent inside some containers
        return [f"unavailable: {e}"]


def collect(box):
    import torch

    info = {
        "box": box,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu": platform.processor() or _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
        "laya_trt_max_batch": os.environ.get("LAYA_TRT_MAX_BATCH", "256"),
        "packages": {p: _pkg(p) for p in (
            "torch", "transformers", "onnxruntime-gpu", "onnxruntime", "tensorrt-cu13", "tensorrt",
            "laya", "nvidia-ml-py", "numpy", "huggingface_hub", "typesafe-sdk")},
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
            "total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        },
        "nvidia_smi": {
            "driver_version": _smi("driver_version"),
            "mig_mode": _smi("mig.mode.current"),
            "sm_clock_max_mhz": _smi("clocks.max.sm"),
            "sm_clock_now_mhz": _smi("clocks.sm"),
            "power_limit_w": _smi("power.limit"),
            "persistence_mode": _smi("persistence_mode"),
        },
    }
    return info


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    args = ap.parse_args()
    info = collect(args.box)
    out = RESULTS / args.box / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
