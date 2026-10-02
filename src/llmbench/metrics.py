import math
import threading
import time

import psutil


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Need nonempty values and a quantile in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(batches):
    if not batches or any(b["seconds"] <= 0 for b in batches):
        raise ValueError("Need positive batch durations")
    seconds = sum(b["seconds"] for b in batches)
    tokens = sum(b["output_tokens"] for b in batches)
    requests = sum(b["requests"] for b in batches)
    return {
        "batch_latency_p50_ms": percentile([b["seconds"] * 1000 for b in batches], 0.5),
        "batch_latency_p95_ms": percentile([b["seconds"] * 1000 for b in batches], 0.95),
        "output_tokens_per_second": tokens / seconds,
        "requests_per_second": requests / seconds,
        "total_seconds": seconds,
        "total_output_tokens": tokens,
        "total_requests": requests,
        "measured_batches": len(batches),
    }


class MemorySampler:
    """Sample process-tree RSS; GPU metric is whole-device used VRAM.

    RSS can double-count shared pages; it is not model size or Apple GPU memory.
    NVML includes other processes, so use an otherwise idle GPU.
    """

    def __init__(self, gpu_index=None, interval=0.02):
        self.interval = interval
        self.stop_event = threading.Event()
        self.rss = 0
        self.gpu = None
        self.nvml = None
        self.error = None
        if gpu_index is not None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.nvml = pynvml
            except Exception as exc:
                self.error = str(exc)

    def sample(self):
        total = 0
        try:
            processes = [psutil.Process()] + psutil.Process().children(recursive=True)
        except psutil.Error:
            processes = [psutil.Process()]
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        self.rss = max(self.rss, total)
        if self.nvml:
            try:
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                self.gpu = max(self.gpu or 0, used)
            except Exception as exc:
                self.error = str(exc)

    def loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.started = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        if self.nvml:
            self.nvml.nvmlShutdown()

    def result(self):
        return {
            "peak_process_tree_rss_mib": self.rss / 2**20,
            "peak_device_used_mib": None if self.gpu is None else self.gpu / 2**20,
            "sample_interval_seconds": self.interval,
            "gpu_sampling_error": self.error,
        }
