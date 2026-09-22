"""Upstream calibration + answer formatting, applied to logits from any backend.

Mirrors `laya.agent.Agent.system_one` line for line (temperature bucket, softmax, confidence,
answer schema) so a backend swap changes nothing but the forward pass.
"""
from typing import Dict, List

import numpy as np
from laya.common import confidence_from_probs, temp_bucket


def calibrate(agent, logits: np.ndarray, meta: List[dict]) -> List[np.ndarray]:
    """Per-row calibrated probability vectors (length k), identical to upstream."""
    out = []
    for r, m in enumerate(meta):
        k, qt = m["k"], m["qtype"]
        scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        z = np.asarray(logits[r, :k], dtype=np.float64) / max(1e-3, float(scale))
        p = np.exp(z - z.max())
        out.append(p / p.sum())
    return out


def format_answers(probs: List[np.ndarray], act: np.ndarray, meta: List[dict], n_requests: int) -> List[Dict]:
    """Upstream answer dicts, grouped back per request."""
    answers = [dict() for _ in range(n_requests)]
    for r, (p, m) in enumerate(zip(probs, meta)):
        q, k = m["q"], m["k"]
        conf = round(confidence_from_probs(p, k), 4)
        ext = {"act_probability": round(float(act[r, 0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            a = {"type": "choice", "choice": keys[int(p.argmax())],
                 "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                 "confidence": conf, "action": ext}
        elif q["t"] == "score":
            a = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                 "legend": {str(i): c for i, c in enumerate(q["crit"])},
                 "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                 "confidence": conf, "action": ext}
        else:
            a = {"type": "noul", "noul": round(float(p[1]), 4),
                 "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
        answers[m["req"]][m["qid"]] = a
    return answers
