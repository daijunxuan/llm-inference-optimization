# Local measurements and interpretation

## Setup and evidence

Apple M2, 8 GiB unified physical memory, macOS 13.3 (Darwin 22.4.0), CPU inference with 4 PyTorch threads. Python 3.13.2, PyTorch 2.6.0, Transformers 4.51.3. Native host execution. Model weights: Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775`.

All four profiles used 8 identical prompts, exactly 64 input tokens and 32 generated tokens, one full warmup batch, and three repeats. There are 768 measured generated tokens per profile. Runs were executed sequentially in separate processes, FP32 B1, FP32 B4, INT8 B1, then INT8 B4.

Raw observations: [run-01](../results/local/run-01/). Derived table: [comparison.csv](../results/local/comparison.csv). Original batch records, software versions and workload hashes remain available for inspection.

## Observations

| Change | Output throughput | Observed ratio |
|---|---|---|
| FP32: B1 → B4 | 12.59 → 20.97 tok/s | 1.66× |
| INT8: B1 → B4 | 18.76 → 56.97 tok/s | 3.04× |
| B4: FP32 → INT8 | 20.97 → 56.97 tok/s | 2.72× |

At B4, sampled peak process-tree RSS decreased from 2,032.9 MiB to 667.2 MiB, approximately 67.2%. This is the observed process memory difference, not a claim that every tensor is compressed by that percentage.

Batching also changes completion time. FP32 B4 takes about 6.10 seconds at P50 versus 1.95 seconds for B1, despite higher total throughput. INT8 B4 completes in about 2.24 seconds versus 1.65 seconds for B1. An application that needs interactive response time should not select a batch size based only on throughput.

## Why these ratios are provisional

The FP32 B1 run has noticeable late-run slowdowns: batch times range from roughly 1.69 to 6.15 seconds. Its P95 is 5.92 seconds versus a 1.95-second median. FP32 B4 batches range from about 5.41 to 6.59 seconds. INT8 B4 is steadier in this small sample, ranging from 2.21 to 2.31 seconds.

The host was an interactive 8 GiB laptop. Background activity, memory pressure and thermal/power behavior may explain some variability, but this experiment did not instrument those causes, so none is asserted. The full original data are retained; outliers were not removed. Profile order was fixed and only one suite was collected. Quantization changes both arithmetic and memory traffic, and these measurements cannot isolate their contributions.

The most defensible conclusion is that **this run observed useful batching and INT8 gains**, motivating more controlled repetitions. It does not establish a universal 2.72× speedup. Before publishing a performance guarantee, repeat the suite in randomized profile order on an idle host, increase warmup and repeats, and report run-to-run uncertainty.

## Validation boundary

- The actual pretrained Qwen model ran through FP32 and INT8 generation in all four profiles.
- Automated tests check metric arithmetic, incompatible-cohort rejection, report preservation, deterministic workloads and real generation through a tiny randomly initialized Qwen model, including INT8 conversion.
- GPU vLLM/NF4 and the Docker image are implemented but not executed here. The local Docker daemon was unavailable and this host has no NVIDIA GPU.
- Sample completions are debugging artifacts. Quantized-model accuracy was not evaluated; no quality parity is claimed.
