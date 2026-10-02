import os
import subprocess
import sys


def test_invalid_batch_rejected_before_loading_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmbench",
            "run",
            "--requests",
            "3",
            "--batch-size",
            "2",
            "--out",
            str(tmp_path / "unused.json"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 2
    assert "divisible" in result.stderr
    assert not (tmp_path / "unused.json").exists()


def test_nonpositive_workload_rejected(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmbench",
            "run",
            "--repeats",
            "0",
            "--out",
            str(tmp_path / "unused.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "positive" in result.stderr
