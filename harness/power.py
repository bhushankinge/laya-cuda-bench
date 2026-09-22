"""Board power / SM clock sampler via NVML, 100 ms default (nvidia-smi dmon floors at 1 s).

    with Sampler() as s: ...work...
    s.stats() -> {"watts_mean", "watts_max", "sm_mhz_mean", "vram_used_mb_max", "samples", "interval_s"}

On a MIG slice NVML reports the parent card's power: the number is per physical GPU, not per slice.
"""
import threading
import time

import pynvml


class Sampler:
    def __init__(self, index=0, interval=0.1):
        self.interval, self.index = interval, index
        self.samples = []  # (t, watts, sm_mhz, mem_used_mb)
        self._stop = threading.Event()

    def __enter__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                clk = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20
                self.samples.append((time.perf_counter(), w, clk, mem))
            except pynvml.NVMLError:  # e.g. power not readable on some laptop parts: record nothing, don't crash
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)
        pynvml.nvmlShutdown()

    def stats(self, since=None):
        s = [x for x in self.samples if since is None or x[0] >= since]
        if not s:
            return {"samples": 0, "interval_s": self.interval}
        return {"watts_mean": sum(x[1] for x in s) / len(s), "watts_max": max(x[1] for x in s),
                "sm_mhz_mean": sum(x[2] for x in s) / len(s), "vram_used_mb_max": max(x[3] for x in s),
                "samples": len(s), "interval_s": self.interval}


if __name__ == "__main__":  # self-check: sampler sees the card and samples at roughly the requested rate
    with Sampler(interval=0.05) as s:
        time.sleep(1.0)
    st = s.stats()
    print(st)
    assert st["samples"] >= 10, st
