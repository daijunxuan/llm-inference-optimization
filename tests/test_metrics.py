import pytest

from llmbench.metrics import MemorySampler, percentile, summarize


def test_throughput_uses_total_tokens_over_total_time():
    # Mean of batch rates would incorrectly give 30 tokens/s.
    value = summarize(
        [
            {"seconds": 1, "output_tokens": 40, "requests": 2},
            {"seconds": 3, "output_tokens": 60, "requests": 3},
        ]
    )
    assert value["output_tokens_per_second"] == 25
    assert value["requests_per_second"] == 1.25
    assert value["batch_latency_p50_ms"] == 2000
    assert value["batch_latency_p95_ms"] == 2900


@pytest.mark.parametrize("values,q,expected", [([7], 0.95, 7), ([4, 1, 2], 0.5, 2)])
def test_percentiles(values, q, expected):
    assert percentile(values, q) == expected


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_memory_without_gpu_does_not_invent_vram():
    with MemorySampler() as sampler:
        pass
    assert sampler.result()["peak_process_tree_rss_mib"] > 0
    assert sampler.result()["peak_device_used_mib"] is None
