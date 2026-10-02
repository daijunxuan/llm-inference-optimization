import csv
import json
from pathlib import Path


def load_results(paths):
    rows = [json.loads(Path(path).read_text()) for path in paths]
    if not rows or any(r.get("status") != "measured" for r in rows):
        raise ValueError("Reports accept measured results only")

    # Batch and precision vary intentionally; workload, hardware and timing policy must match.
    def cohort(r):
        c, e = r["config"], r["environment"]
        return (
            c["model"],
            c["revision"],
            c["device"],
            c["input_tokens"],
            c["output_tokens"],
            c["requests"],
            c["repeats"],
            c["warmup"],
            c["seed"],
            c["threads"],
            r["workload_sha256"],
            e["os"],
            e["os_release"],
            e["architecture"],
            e["device_name"],
        )

    if len({cohort(r) for r in rows}) != 1:
        raise ValueError("Incompatible benchmark cohorts; report each hardware/workload separately")
    return rows


def report(args):
    results = load_results(args.files)
    results.sort(
        key=lambda r: (
            {("hf", "none"): 0, ("vllm", "none"): 1, ("hf", "int8"): 2, ("hf", "nf4"): 3}.get(
                (r["config"]["backend"], r["config"]["quantization"]), 4
            ),
            r["config"]["batch_size"],
        )
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = [
        "| Configuration | Batch | P50 batch latency (ms) | P95 (ms) | Output tok/s | Peak RSS (MiB) | Device VRAM (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for r in results:
        c, m, mem = r["config"], r["metrics"], r["memory"]
        precision = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}[c["dtype"]]
        if c["quantization"] == "int8":
            label = "HF dynamic INT8"
        elif c["quantization"] == "nf4":
            label = f"HF NF4 ({precision} compute)"
        else:
            label = f"{'HF baseline' if c['backend'] == 'hf' else 'vLLM'} {precision}"
        vram = mem["peak_device_used_mib"]
        values = [
            label,
            c["batch_size"],
            f"{m['batch_latency_p50_ms']:.2f}",
            f"{m['batch_latency_p95_ms']:.2f}",
            f"{m['output_tokens_per_second']:.2f}",
            f"{mem['peak_process_tree_rss_mib']:.1f}",
            "N/A" if vram is None else f"{vram:.1f}",
        ]
        table.append("| " + " | ".join(map(str, values)) + " |")
        csv_rows.append(
            dict(
                zip(
                    [
                        "configuration",
                        "batch_size",
                        "p50_ms",
                        "p95_ms",
                        "output_tok_s",
                        "peak_rss_mib",
                        "device_vram_mib",
                    ],
                    values,
                )
            )
        )
    markdown = "\n".join(table) + "\n"
    (out / "comparison.md").write_text(markdown)
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    if args.readme:
        path = Path(args.readme)
        contents = path.read_text()
        start, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
        if contents.count(start) != 1 or contents.count(end) != 1:
            raise ValueError("README must contain one pair of benchmark markers")
        before, rest = contents.split(start)
        _, after = rest.split(end)
        path.write_text(before + start + "\n" + markdown + end + after)
    print(markdown)
