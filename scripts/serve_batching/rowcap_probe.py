"""Row-cap probe: LAYA_BATCH_MAX counts requests, but the forward pass holds requests x questions rows.

    python rowcap_probe.py <port> <n_concurrent> <n_questions>
Sends n_concurrent identical-schema requests (long states, n_questions noul questions each) at once and
prints wall time, per-request X-Inference-Time-Ms and status codes.
"""
import asyncio
import os
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.workload import load_workload  # noqa: E402

port, n_conc, n_q = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
wl = load_workload("laya")
states = wl["buckets"]["long"]
questions = {"q%02d" % i: {"type": "noul", "instructions": "Does the notice mention requirement number %d?" % i}
             for i in range(n_q)}


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        # one warm request of this shape first, so the timed run is not the first forward of that size
        async with s.post(f"http://127.0.0.1:{port}/v1/systemone", json={"state": states[0], "questions": questions}) as r:
            await r.read()
        t = time.perf_counter()
        rs = await asyncio.gather(*[s.post(f"http://127.0.0.1:{port}/v1/systemone",
                                           json={"state": states[i % len(states)], "questions": questions})
                                    for i in range(n_conc)])
        wall = (time.perf_counter() - t) * 1e3
        for r in rs:
            await r.read()
        infer = sorted(float(r.headers.get("X-Inference-Time-Ms", "nan")) for r in rs)
        print(f"n_conc={n_conc} n_q={n_q} rows={n_conc * n_q} wall={wall:.0f}ms status={sorted(set(r.status for r in rs))} "
              f"infer_ms min/max={infer[0]:.0f}/{infer[-1]:.0f}")

asyncio.run(main())
