"""Multi-state batching on top of upstream laya prompt construction.

Upstream `Agent.predict` batches the questions of ONE state. Here a batch is any list of
(state, questions) requests; every (state, question) pair becomes one row, built with the
upstream `build_sequence` so tokens are identical to what `Agent.predict` would produce.
"""
from typing import Any, Dict, List, Tuple

import torch
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, collate_items, render_options

from . import MODELS

Request = Tuple[Any, Dict[str, Dict[str, Any]]]  # (state, questions)


def load_agent(name: str, device: str = "cuda") -> Agent:
    """Load a checkpoint from models/<name> (downloaded by hf download)."""
    return Agent(str(MODELS / name), device=device)


def build_rows(agent: Agent, requests: List[Request]):
    """Return (items, meta). items feed upstream collate_items; meta maps rows back to requests."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items, meta = [], []
    for r, (state, questions) in enumerate(requests):
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            meta.append({"req": r, "qid": qid, "q": q, "k": len(markers), "qtype": QTYPES[q["t"]]})
    return items, meta


def collate(agent: Agent, items, pad_to: int = None) -> Dict[str, torch.Tensor]:
    """Upstream collate, optionally right-padded to a fixed length (length buckets, TRT profiles)."""
    b = collate_items([items], agent.tok.pad_token_id)
    if pad_to is not None and b["input_ids"].shape[1] < pad_to:
        n, L = b["input_ids"].shape
        extra = pad_to - L
        b["input_ids"] = torch.cat([b["input_ids"], torch.full((n, extra), agent.tok.pad_token_id, dtype=torch.long)], 1)
        b["attention_mask"] = torch.cat([b["attention_mask"], torch.zeros((n, extra), dtype=torch.long)], 1)
    return {k: b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
