"""Export a checkpoint to ONNX (FP32) and derive an FP16 copy for ONNX Runtime CUDA/TensorRT.

    python -m harness.export_onnx --models laya laya-multilingual laya-typed-decisions

Graph: inputs input_ids[B,L] i64, attention_mask[B,L] i64, marker_pos[B,K] i64, marker_mask[B,K] bool,
qtype[B] i64 -> logits[B,K] f32 (masked slots -1e4), act_probs[B,2] f32. Same wrapper receptron/laya
uses (dynamo exporter, opset 18, dynamic batch/seq/options). Export runs on CPU in FP32 so the graph is
device-independent; a CPU parity check on one padded batch is printed.
"""
import argparse
import time

import numpy as np
import torch

from . import CHECKPOINTS, MODELS
from .sequences import load_agent


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, torch.softmax(act.float(), -1)


def example(B=2, L=40, K=4):
    ex = (torch.randint(5, 1000, (B, L)), torch.ones(B, L, dtype=torch.long),
          torch.tensor([[3, 9, 15, 21], [3, 9, 0, 0]]), torch.tensor([[True] * 4, [True, True, False, False]]),
          torch.tensor([0, 2]))
    ex[1][1, 30:] = 0
    return ex


def export(name):
    agent = load_agent(name, device="cpu")
    w = Wrapper(agent.model.float().eval())
    out_dir = MODELS / name / "onnx"
    out_dir.mkdir(exist_ok=True)
    fp32 = out_dir / f"{name}-fp32.onnx"
    ex = example()
    batch, seq, opts = torch.export.Dim("batch"), torch.export.Dim("seq", min=8), torch.export.Dim("options", min=2)
    t0 = time.perf_counter()
    # grad must stay enabled: nn.TransformerEncoderLayer's fused fast path (not exportable) is only taken under no_grad
    prog = torch.onnx.export(
        w, ex, opset_version=18, dynamo=True, optimize=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_probs"],
        dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
                        "marker_pos": {0: batch, 1: opts}, "marker_mask": {0: batch, 1: opts}, "qtype": {0: batch}},
    )
    prog.save(str(fp32), external_data=False)
    export_s = time.perf_counter() - t0

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m16 = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
    fp16 = out_dir / f"{name}-fp16.onnx"
    onnx.save(m16, str(fp16))

    import onnxruntime as ort
    with torch.no_grad():
        ref_logits, ref_act = w(*ex)
    feed = dict(zip(["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"], (t.numpy() for t in ex)))
    o = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(None, feed)
    print(f"{name}: exported in {export_s:.0f}s -> {fp32.name} ({fp32.stat().st_size/1e6:.0f} MB), {fp16.name} "
          f"({fp16.stat().st_size/1e6:.0f} MB); CPU fp32 parity max|dlogits|={np.abs(o[0]-ref_logits.numpy()).max():.2e} "
          f"max|dact|={np.abs(o[1]-ref_act.numpy()).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CHECKPOINTS))
    for name in ap.parse_args().models:
        export(name)


if __name__ == "__main__":
    main()
