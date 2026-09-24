"""Stock forward: bf16 autocast (the cc>=8 default) vs fp16 autocast vs fp32, on benchmarks/parity_fast.py's fixed set.

    LAYA_SRC=<laya checkout> python scripts/dtype_parity.py <checkpoint dir>
"""
import os, sys, time, json
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1"); os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
UP = os.environ.get("LAYA_SRC", os.path.expanduser("~/laya"))  # a checkout of NandhaKishorM/laya (needs benchmarks/parity_fast.py)
sys.path.insert(0, UP); sys.path.insert(0, UP + "/benchmarks")
import numpy as np, torch, laya
import parity_fast as pf

path = sys.argv[1]
agent = laya.load(path)
assert agent.device.type == "cuda"
default = agent.dtype
gpu = torch.cuda.get_device_name(0)
by_t, cases = {}, []
for name, state, qs in pf.states():
    b, meta = pf.batch(agent, state, qs)
    p32 = pf.probs(agent, b, meta, amp=False)
    agent.dtype = torch.bfloat16; pbf = pf.probs(agent, b, meta, amp=True)
    agent.dtype = torch.float16; pfp = pf.probs(agent, b, meta, amp=True)
    for (qid, t, k), x, y, z in zip(meta, p32, pbf, pfp):
        x, y, z = map(np.array, (x, y, z))
        d = by_t.setdefault(t, dict(n=0, bf16=0.0, fp16=0.0, agree_bf16=0, agree_fp16=0))
        d["n"] += 1; d["bf16"] = max(d["bf16"], float(abs(y - x).max())); d["fp16"] = max(d["fp16"], float(abs(z - x).max()))
        d["agree_bf16"] += int(y.argmax() == x.argmax()); d["agree_fp16"] += int(z.argmax() == x.argmax())
        cases.append(dict(state=name, question=qid, type=t, k=k, p_fp32=x.tolist(), p_bf16=y.tolist(), p_fp16=z.tolist()))
agent.dtype = default
print(f"\n{os.path.basename(path)}  {gpu}  torch {torch.__version__}  default dtype {default}  cc {torch.cuda.get_device_capability()}")
print(f"{'type':8s} {'n':>4s} {'max|bf16-fp32|':>15s} {'max|fp16-fp32|':>15s} {'argmax bf16=fp32':>17s} {'fp16=fp32':>10s}")
for t, d in by_t.items():
    print(f"{t:8s} {d['n']:4d} {d['bf16']:15.4f} {d['fp16']:15.4f} {d['agree_bf16']:>12d}/{d['n']:<4d} {d['agree_fp16']:>6d}/{d['n']}")
allc = [(abs(np.array(c['p_bf16'])-np.array(c['p_fp32'])).max(), abs(np.array(c['p_fp16'])-np.array(c['p_fp32'])).max()) for c in cases]
print(f"all      {len(cases):4d} {max(a for a,_ in allc):15.4f} {max(b for _,b in allc):15.4f}   median {np.median([a for a,_ in allc]):.4f} vs {np.median([b for _,b in allc]):.4f}")

# latency: one short state x triage questions, end-to-end predict
state = {"subject": pf.TEXTS[1][:60], "body": pf.TEXTS[1]}
qs = dict(list(pf.PRESETS["triage"]().items())[:3])
lat = {}
for dt in (torch.bfloat16, torch.float16):
    agent.dtype = dt
    for _ in range(20): agent.predict(state, qs)
    torch.cuda.synchronize(); ts = []
    for _ in range(100):
        t0 = time.perf_counter(); agent.predict(state, qs); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    lat[str(dt)] = dict(p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)))
agent.dtype = default
print("latency ms (3 questions, 100 iters):", json.dumps(lat))
json.dump(dict(model=os.path.basename(path), gpu=gpu, torch=torch.__version__, default_dtype=str(default), summary=by_t, latency=lat, cases=cases), open(f"dtype_parity_{os.path.basename(path)}.json", "w"), indent=1)
