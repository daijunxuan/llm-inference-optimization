"""Plot the committed local results using their original measurement definitions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
names = ["hf-none-b1", "hf-none-b4", "hf-int8-b1", "hf-int8-b4"]
rows = [json.loads((ROOT / "results/local/run-01" / f"{n}.json").read_text()) for n in names]
labels = ["FP32 / B1", "FP32 / B4", "INT8 / B1", "INT8 / B4"]
colors = ["#94a3b8", "#64748b", "#2dd4bf", "#0d9488"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
panels = [
    (
        "Output throughput",
        "Output tokens / second",
        [r["metrics"]["output_tokens_per_second"] for r in rows],
    ),
    (
        "P50 batch completion",
        "Milliseconds / batch",
        [r["metrics"]["batch_latency_p50_ms"] for r in rows],
    ),
    (
        "Sampled host memory",
        "Peak process-tree RSS / MiB",
        [r["memory"]["peak_process_tree_rss_mib"] for r in rows],
    ),
]
for ax, (title, ylabel, values) in zip(axes, panels):
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.bar_label(bars, labels=[f"{v:,.1f}" for v in values], padding=5)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(axis="x", rotation=20)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.6)
fig.suptitle(
    "Qwen 0.5B · Apple M2 CPU · Actual local measurements",
    x=0.06,
    ha="left",
    fontsize=17,
    fontweight="bold",
)
fig.text(
    0.06,
    0.025,
    "64 input / 32 output tokens · 4 threads · 3 repeats · "
    "short laptop run; variability and raw data documented",
    color="#475569",
    fontsize=10,
)
fig.tight_layout(rect=(0.02, 0.07, 1, 0.91))
fig.savefig(ROOT / "results/local/comparison.png", dpi=180, facecolor="white")
