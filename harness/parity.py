"""Correctness gate: every backend must reproduce upstream on the 63 laya-mlx parity questions.

Reference = upstream `laya` model run in FP32 on this GPU (no autocast), exactly as the MLX port's
validate.py did on MPS. Gate: 63/63 argmax agreement for every backend; FP32 backends max-abs
probability error <= 1e-4. FP16/BF16 error is reported and flagged above 1e-3 (the port gated at 0.02).
Also: 100 repeated calls must not grow torch.cuda.memory_allocated() by more than 1 MiB.

    python -m harness.parity --box laptop-rtx2000ada --backends eager-fp32 eager-fp16 eager-bf16 \
        [ort-cuda-fp32 ort-cuda-fp16 ort-trt-fp16 compile-fp16] [--models laya ...]
"""
import argparse
import gc
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch

from . import CHECKPOINTS, FIXTURES, MODELS, RESULTS
from .backends import ort_backend, torch_compile, torch_eager
from .predict import calibrate, format_answers
from .sequences import build_rows, collate, load_agent, to_device

FP32_TOL, FP16_FLAG = 1e-4, 1e-3
GROWTH_TOL = 1 << 20  # bytes over 100 calls; the port used 32 MiB. Readings jitter by a few KB with what is referenced.


def load_cases():
    return json.loads((FIXTURES / "parity" / "cases.json").read_text())


def make_backend(name, agent, model_name):
    kind, *rest = name.split("-")
    # ORT holds its own copy of the weights; park the torch model on the CPU so an 8 GB card fits both paths.
    agent.model.to("cpu" if kind == "ort" else agent.device)
    torch.cuda.empty_cache()
    if kind == "eager":
        return torch_eager(agent.model, rest[0])
    if kind == "compile":
        return torch_compile(agent.model, rest[0])
    if kind == "ort":
        provider, dtype = rest
        onnx = MODELS / model_name / f"onnx/{model_name}-{dtype}.onnx"
        kw = {"tf32": dtype != "fp32"}  # an FP32 reference row must not silently run TF32
        if provider == "trt":
            kw = {"trt_cache": MODELS / model_name / "trt_cache", "trt_profile": trt_profile(agent)}
        return ort_backend(onnx, provider, **kw)
    raise ValueError(name)


def trt_profile(agent, max_batch=None):
    """TensorRT optimisation profile. The max shape sets engine-build workspace: 256x512 needs ~10 GB, so small
    cards set LAYA_TRT_MAX_BATCH (e.g. 32 on an 8 GB laptop); the recorded backend name carries the value."""
    import os
    max_batch = max_batch or int(os.environ.get("LAYA_TRT_MAX_BATCH", 256))
    L = agent.cfg.get("max_len", 512)
    shp = lambda b, l, k: f"input_ids:{b}x{l},attention_mask:{b}x{l},marker_pos:{b}x{k},marker_mask:{b}x{k},qtype:{b}"
    return {"min": shp(1, 8, 2), "opt": shp(16, 256, 4), "max": shp(max_batch, L, 20)}


def reference_outputs(agent, cases):
    """FP32 upstream logits/act per case + upstream public predict() result (its own default dtype)."""
    agent.model.float()
    out = []
    for c in cases:
        items, meta = build_rows(agent, [(c["state"], c["questions"])])
        b = to_device(collate(agent, items), agent.device)
        with torch.inference_mode():
            logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        torch.cuda.synchronize()
        logits, act = logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()
        out.append({"case": c["name"], "items": items, "meta": meta, "logits": logits, "act": act,
                    "probs": calibrate(agent, logits, meta), "public": agent.predict(c["state"], c["questions"])})
    return out


def check_backend(agent, fn, refs, repeats):
    agree = total = 0
    max_p_err = max_act_err = 0.0
    public_equal = 0
    rows = []
    for r in refs:
        b = to_device(collate(agent, r["items"]), agent.device)
        logits, act = fn(b)
        assert np.isfinite(logits).all() and np.isfinite(act).all(), r["case"]
        probs = calibrate(agent, logits, r["meta"])
        p_err = max(float(np.max(np.abs(a - b_))) for a, b_ in zip(probs, r["probs"]))
        a_err = float(np.max(np.abs(act - r["act"])))
        ag = sum(int(a.argmax() == b_.argmax()) for a, b_ in zip(probs, r["probs"]))
        answers = format_answers(probs, act, r["meta"], 1)[0]
        public_equal += int(answers == r["public"]["answers"])
        rows.append({"case": r["case"], "questions": len(r["meta"]), "argmax_agreements": ag,
                     "probability_max_abs_error": p_err, "action_probability_max_abs_error": a_err,
                     "public_answers_equal": answers == r["public"]["answers"]})
        agree += ag; total += len(r["meta"]); max_p_err = max(max_p_err, p_err); max_act_err = max(max_act_err, a_err)
    # stability: repeated identical calls, zero steady-state allocator growth (measured after one warm call,
    # because the first call parks ~1 KB of autocast/workspace state that never grows again)
    fn(to_device(collate(agent, refs[0]["items"]), agent.device))
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    first = None
    t0 = time.perf_counter()
    for i in range(repeats):
        r = refs[i % len(refs)]
        b = to_device(collate(agent, r["items"]), agent.device)
        logits, _ = fn(b)
        d = np.asarray([p.argmax() for p in calibrate(agent, logits, r["meta"])])
        key = (r["case"], d.tobytes())
        first = first or {}
        assert first.setdefault(r["case"], key) == key, f"nondeterministic argmax on call {i}"
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del b, logits  # the last input batch is still referenced here and would read as ~1 KB of "growth"
    gc.collect(); torch.cuda.empty_cache()
    growth = torch.cuda.memory_allocated() - before
    return {"backend": fn.name, "argmax_agreements": agree, "questions": total,
            "probability_max_abs_error": max_p_err, "action_probability_max_abs_error": max_act_err,
            "public_answers_equal_cases": public_equal, "cases": rows,
            "stability": {"calls": repeats, "elapsed_seconds": elapsed, "allocated_growth_bytes": int(growth),
                          "peak_bytes": torch.cuda.max_memory_allocated()},
            "compile_s": getattr(fn, "compile_s", None), "session_build_s": getattr(fn, "session_build_s", None)}


def verdict(rep, backend):
    fp32 = backend.endswith("fp32")
    ok = rep["argmax_agreements"] == rep["questions"] and rep["stability"]["allocated_growth_bytes"] <= GROWTH_TOL
    if fp32:
        ok = ok and rep["probability_max_abs_error"] <= FP32_TOL
    flag = (not fp32) and rep["probability_max_abs_error"] > FP16_FLAG
    return ("PASS" if ok else "FAIL") + (" (fp16 error > 1e-3, port gated at 0.02)" if flag and ok else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", required=True)
    ap.add_argument("--models", nargs="+", default=list(CHECKPOINTS))
    ap.add_argument("--backends", nargs="+", default=["eager-fp32", "eager-fp16", "eager-bf16"])
    ap.add_argument("--repeats", type=int, default=100)
    args = ap.parse_args()
    cases = load_cases()
    out_path = RESULTS / args.box / "parity.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {"results": []}
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["gpu"] = torch.cuda.get_device_name(0)
    failures = 0
    for name in args.models:
        agent = load_agent(name)
        refs = reference_outputs(agent, cases)
        for backend in args.backends:
            fn = make_backend(backend, agent, name)
            rep = check_backend(agent, fn, refs, args.repeats)
            rep.update({"model": name, "verdict": verdict(rep, backend)})
            failures += rep["verdict"].startswith("FAIL")
            report["results"] = [r for r in report["results"] if not (r["model"] == name and r["backend"] == rep["backend"])] + [rep]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=1) + "\n")  # crash-safe: a later engine-build OOM must not lose rows
            print(f"{name:22s} {rep['backend']:28s} {rep['argmax_agreements']}/{rep['questions']} argmax  "
                  f"p_err={rep['probability_max_abs_error']:.3g} act_err={rep['action_probability_max_abs_error']:.3g} "
                  f"growth={rep['stability']['allocated_growth_bytes']}B  {rep['verdict']}", flush=True)
            del fn; gc.collect(); torch.cuda.empty_cache()
        del agent, refs; gc.collect(); torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", out_path, "| failures:", failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
