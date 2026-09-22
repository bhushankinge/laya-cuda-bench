"""Forward backends. Each `make_*` returns fn(batch_on_device) -> (logits[n,k] f32 numpy, act_probs[n,2] f32 numpy).

The returned callable blocks until the result is on the host, so timing around it includes the
device->host copy, matching the port's timing boundary.
"""
import time
from pathlib import Path

import numpy as np
import torch

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def torch_eager(model, dtype: str = "fp16"):
    dt = DTYPES[dtype]
    use_amp = dtype != "fp32"

    @torch.inference_mode()
    def fn(b):
        with torch.autocast(device_type="cuda", dtype=dt, enabled=use_amp):
            logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        act = torch.softmax(act.float(), -1)
        return logits.float().cpu().numpy(), act.cpu().numpy()

    fn.name = f"torch-eager-{dtype}"
    return fn


def torch_compile(model, dtype: str = "fp16", mode: str = "max-autotune"):
    """torch.compile with dynamic shapes; first-call compile time is stored on fn.compile_s."""
    compiled = torch.compile(model, mode=mode, dynamic=True)
    inner = torch_eager(compiled, dtype)

    def fn(b):
        t0 = time.perf_counter()
        out = inner(b)
        if fn.compile_s is None:
            torch.cuda.synchronize()
            fn.compile_s = time.perf_counter() - t0
        return out

    fn.compile_s = None
    fn.name = f"torch-compile-{mode}-{dtype}"
    return fn


def ort_session(onnx_path: Path, provider: str, trt_cache: Path = None, trt_profile: dict = None, tf32: bool = True):
    """provider: 'cuda' or 'trt'. trt_profile: {'min': 'input_ids:1x8,...', 'opt': ..., 'max': ...}.
    tf32=False makes the CUDA EP do true FP32 matmuls (its default TF32 costs ~2.6e-3 probability error vs torch FP32)."""
    import onnxruntime as ort

    if provider == "trt":
        # pip TensorRT lives in site-packages/tensorrt_libs, which is not on the loader path; preload it.
        import ctypes, importlib.util
        libdir = Path(importlib.util.find_spec("tensorrt_libs").origin).parent
        for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            ctypes.CDLL(str(libdir / lib), mode=ctypes.RTLD_GLOBAL)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 1 if tf32 else 0,
                                                "arena_extend_strategy": "kSameAsRequested"})]
    elif provider == "trt":
        opts = {"device_id": 0, "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache or Path("trt_cache")), "trt_timing_cache_enable": True,
                "trt_layer_norm_fp32_fallback": True}  # pure-FP16 LayerNorm put the max probability error at 0.025 (> port's 0.02)
        if trt_profile:
            opts.update({"trt_profile_min_shapes": trt_profile["min"], "trt_profile_opt_shapes": trt_profile["opt"],
                         "trt_profile_max_shapes": trt_profile["max"]})
        providers = [("TensorrtExecutionProvider", opts), ("CUDAExecutionProvider", {"device_id": 0})]
    else:
        raise ValueError(provider)
    sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
    want = {"cuda": "CUDAExecutionProvider", "trt": "TensorrtExecutionProvider"}[provider]
    if sess.get_providers()[0] != want:  # ORT falls back silently; a benchmark row must never mislabel its backend
        raise RuntimeError(f"requested {want} but session runs on {sess.get_providers()}")
    return sess


def ort_backend(onnx_path: Path, provider: str = "cuda", **kw):
    """ONNX Runtime backend. Inputs are fed from host numpy (the graph inputs are tiny int tensors)."""
    t0 = time.perf_counter()
    sess = ort_session(onnx_path, provider, **kw)
    build_s = time.perf_counter() - t0
    names = [i.name for i in sess.get_inputs()]

    def fn(b):
        feed = {n: b[n].cpu().numpy() for n in names}
        logits, act = sess.run(None, feed)
        return np.asarray(logits, dtype=np.float32), np.asarray(act, dtype=np.float32)

    fn.name = f"ort-{provider}-{Path(onnx_path).stem.split('-')[-1]}"
    if provider == "trt" and kw.get("trt_profile"):
        fn.name += "-maxb" + kw["trt_profile"]["max"].split(":")[1].split("x")[0]
    fn.session_build_s = build_s  # TRT engine build happens lazily on first run; see fn.first_run_s in callers
    return fn
