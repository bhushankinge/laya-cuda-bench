"""Minimal serving stack: HTTP /predict with a dynamic batcher over one backend.

    python -m harness.server --model laya --backend eager-fp16 --max-batch 64 --max-delay-ms 2 --port 8080

Same algorithm Triton's dynamic batcher documents: take the first waiting request, keep collecting until
max_batch rows or max_delay elapses, run one forward, answer everyone. One GPU worker thread; the event
loop keeps accepting requests while the forward runs. Response = upstream predict() schema plus
"served_batch" (rows in the forward that answered this request) so the load generator can record it.
"""
import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

from .parity import make_backend
from .predict import calibrate, format_answers
from .sequences import build_rows, collate, load_agent, to_device


class Batcher:
    def __init__(self, agent, fn, max_batch, max_delay_ms):
        self.agent, self.fn, self.max_batch, self.max_delay = agent, fn, max_batch, max_delay_ms / 1e3
        self.q = asyncio.Queue()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.batches_served = 0

    def _forward(self, reqs):
        items, meta = build_rows(self.agent, reqs)
        b = to_device(collate(self.agent, items), self.agent.device)
        logits, act = self.fn(b)
        probs = calibrate(self.agent, logits, meta)
        answers = format_answers(probs, act, meta, len(reqs))
        tokens = b["attention_mask"].sum(1).tolist()
        per_req_tokens = [0] * len(reqs)
        for m, t in zip(meta, tokens):
            per_req_tokens[m["req"]] += int(t)
        return answers, per_req_tokens

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.q.get()
            pending = [first]
            rows = len(first[0][1])
            deadline = loop.time() + self.max_delay
            while rows < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.q.get(), timeout)
                except asyncio.TimeoutError:
                    break
                pending.append(item)
                rows += len(item[0][1])
            reqs = [p[0] for p in pending]
            try:
                answers, toks = await loop.run_in_executor(self.pool, self._forward, reqs)
                self.batches_served += 1
                for (_, fut), a, t in zip(pending, answers, toks):
                    if not fut.done():
                        fut.set_result({"model": "laya-rl-agent", "answers": a,
                                        "usage": {"input_tokens": t, "output_tokens": 0}, "served_batch": rows})
            except Exception as e:  # one bad request must not poison the batch's neighbours silently
                for _, fut in pending:
                    if not fut.done():
                        fut.set_exception(e)


async def predict(request):
    body = await request.json()
    if "questions" not in body or not isinstance(body["questions"], dict):
        raise web.HTTPBadRequest(text="need {state, questions}")
    fut = asyncio.get_running_loop().create_future()
    await request.app["batcher"].q.put(((body.get("state", ""), body["questions"]), fut))
    try:
        return web.json_response(await fut)
    except ValueError as e:
        raise web.HTTPBadRequest(text=str(e))


async def healthz(request):
    return web.json_response({"ok": True, "backend": request.app["backend"], "batches_served": request.app["batcher"].batches_served})


def make_app(model, backend, max_batch, max_delay_ms):
    agent = load_agent(model)
    fn = make_backend(backend, agent, model)
    t0 = time.perf_counter()  # warm the backend once (compile / engine build) before accepting traffic
    from .workload import load_workload
    wl = load_workload(model)
    for n in (1, min(16, max_batch), max_batch):
        Batcher(agent, fn, max_batch, max_delay_ms)._forward([(wl["natural"][i % len(wl["natural"])], wl["questions"]) for i in range(max(1, n // 3))])
    print(f"warmup {time.perf_counter() - t0:.1f}s", flush=True)
    app = web.Application(client_max_size=4 * 2**20)
    app["batcher"], app["backend"] = Batcher(agent, fn, max_batch, max_delay_ms), fn.name

    async def start(app):
        app["task"] = asyncio.create_task(app["batcher"].run())
    app.on_startup.append(start)
    app.router.add_post("/predict", predict)
    app.router.add_get("/healthz", healthz)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya")
    ap.add_argument("--backend", default="eager-fp16")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--max-delay-ms", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    web.run_app(make_app(args.model, args.backend, args.max_batch, args.max_delay_ms), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
