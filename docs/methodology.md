# Measurement protocol

## Scope

This is an offline, bounded-batch generation benchmark. Submit B requests together, wait for the full batch, then submit the next batch. The experiment isolates runtime, batch size, and weight representation under a fixed token budget. It does not simulate independently arriving requests.

1. Start a fresh process per configuration.
2. Resolve the model revision and use it for the tokenizer and model.
3. Build deterministic raw-text token sequences. Repeat/truncate tokens to the exact input length; save the SHA-256 hash of all token IDs.
4. Load the backend and prepare batches. HF input tensors are resident on the target device before timing; vLLM receives token IDs and manages transfers internally.
5. Warm up the first batch the configured number of times. vLLM prefix caching is disabled so repeated trials do not become prefix-cache hits.
6. For every repeat and batch, synchronize if needed, time blocking generation, synchronize again, then stop timing.
7. Extract output IDs outside the timed region and validate exactly the requested number of sequences and output tokens. Greedy decoding, EOS disabled/ignored, and equal output lengths prevent early stopping from changing the workload.
8. Save raw durations, counts, environment, memory samples' peak, and a sample completion.

No chat template is applied: this is a synthetic token workload. Repeated/truncated prompts and forced-length output are intentionally unsuitable for assessing instruction-following quality.

## Metrics

For timed batches i with durations t_i and generated token counts n_i:

```text
output tokens/s = sum(n_i) / sum(t_i)
requests/s     = sum(batch_request_count_i) / sum(t_i)
P50 / P95      = linearly interpolated quantiles of batch durations
```

Batch latency is not divided by B and relabeled as request latency. All requests are submitted at once, and only full-batch completion is observed. P95 based on a handful of batches is descriptive, not a reliable production tail estimate. The local B=4 experiment has only six measured batches; increase repeats substantially for statistical inference.

The timing boundary excludes tokenization, model download/load, warmup, explicit HF input transfer, output extraction, and report generation. It includes HF generation control flow and vLLM enqueue/scheduling overhead. Consequently this is an API-level offline experiment, not a pure kernel microbenchmark.

## Memory interpretation

The memory sampler runs during measured generation, every 20 ms, with additional start/end observations:

- **Process-tree RSS:** sums resident bytes of the driver and child processes. Shared pages may be counted more than once. It includes resident weights, Python/runtime allocations and allocator caches. It is not peak model-loading memory, tensor size, or Apple Metal memory.
- **NVML device used:** whole-device used VRAM, including KV-cache reservation, CUDA runtime and other processes. It is not a per-process allocation measure. Use an idle, dedicated GPU. vLLM often reserves most of its configured GPU budget, so NF4 weight savings need not produce a proportionate device-memory reduction.
- **Sampling:** brief peaks between observations can be missed. Memory monitoring adds some overhead to every backend.
- **Unavailable values:** represented as null/N/A, never as zero. GPU sampler errors are preserved in the result.

Use `--nvml-index` for the **physical** NVIDIA device matching the CUDA-visible device. Container GPU visibility can remap CUDA IDs; verify the NVML mapping for your deployment. This implementation supports one GPU only and does not validate MIG partitions.

## Comparing experiments fairly

The report rejects differences in model/revision, device, OS, CPU thread count, workload hash, input/output lengths, request count, repeats, warmup and seed. Batch and quantization settings may vary. Package versions are recorded because different engine stacks can constrain dependency choices. Run the complete GPU matrix inside one image for a stronger comparison.

Hardware metadata cannot prove two identical GPU model names refer to the same physical GPU. Run comparisons on the same dedicated host; record host hardware, driver version, thermal state and power policy with your publication. Avoid unrelated workloads while measuring. The first local run was made on an interactive laptop, so OS background activity is a source of noise.

## Quantization and quality

- CPU INT8 dynamically quantizes Linear modules using PyTorch; weights of unsupported modules remain FP32. ARM uses QNNPACK when available. Conversion is in place before measurements.
- GPU NF4 uses bitsandbytes with double quantization and FP16 compute. It is a separate CUDA experiment, not the same algorithm as dynamic INT8.
- vLLM uses FP16 with prefix caching disabled. Its GPU utilization budget is 0.8 by default; that reservation affects memory comparisons.
- Every run records one completion for debugging. That sample is not an accuracy benchmark. No model-quality equivalence, lossless compression, or production recommendation is implied by throughput measurements.

Before deploying a quantized model, evaluate held-out task accuracy/perplexity, output format compliance, and safety behavior on the actual application distribution. Keep those scores alongside performance data, with their own datasets and protocols.
