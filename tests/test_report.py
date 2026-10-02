import copy
import json
from types import SimpleNamespace

import pytest

from llmbench.report import load_results, report


@pytest.fixture
def result():
    return {
        "status": "measured",
        "config": {
            "model": "test",
            "revision": "abc",
            "device": "cpu",
            "input_tokens": 64,
            "output_tokens": 32,
            "requests": 8,
            "repeats": 3,
            "warmup": 1,
            "seed": 42,
            "threads": 4,
            "backend": "hf",
            "quantization": "none",
            "dtype": "float32",
            "batch_size": 1,
        },
        "workload_sha256": "xyz",
        "environment": {
            "os": "test",
            "os_release": "1",
            "architecture": "test",
            "device_name": "test",
        },
        "metrics": {
            "batch_latency_p50_ms": 100,
            "batch_latency_p95_ms": 120,
            "output_tokens_per_second": 320,
        },
        "memory": {"peak_process_tree_rss_mib": 42, "peak_device_used_mib": None},
    }


def write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize(
    "field,value",
    [("device", "cuda"), ("revision", "other"), ("output_tokens", 99), ("threads", 1)],
)
def test_reject_mismatched_cohorts(tmp_path, result, field, value):
    other = copy.deepcopy(result)
    other["config"][field] = value
    with pytest.raises(ValueError, match="Incompatible"):
        load_results([write(tmp_path, "a.json", result), write(tmp_path, "b.json", other)])


def test_accept_batch_and_precision_variation(tmp_path, result):
    other = copy.deepcopy(result)
    other["config"].update(batch_size=4, quantization="int8")
    assert (
        len(load_results([write(tmp_path, "a.json", result), write(tmp_path, "b.json", other)]))
        == 2
    )


def test_no_fake_results(tmp_path, result):
    result["status"] = "simulated"
    with pytest.raises(ValueError, match="measured"):
        load_results([write(tmp_path, "a.json", result)])


def test_report_and_readme_preservation(tmp_path, result):
    path = write(tmp_path, "a.json", result)
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- BENCHMARK:START -->\nold\n<!-- BENCHMARK:END -->\nafter")
    report(SimpleNamespace(files=[path], out=tmp_path / "report", readme=readme))
    assert "| 320.00 | 42.0 | N/A |" in readme.read_text()
    assert readme.read_text().startswith("before\n")
    assert readme.read_text().endswith("\nafter")
    assert (tmp_path / "report/comparison.csv").exists()
