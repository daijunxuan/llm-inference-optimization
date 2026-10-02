import argparse

from .workload import MODEL, REVISION


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Reproducible offline LLM inference experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Measure one model in a fresh process")
    run.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    run.add_argument("--model", default=MODEL)
    run.add_argument("--revision", default=REVISION)
    run.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    run.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    run.add_argument("--quantization", choices=["none", "int8", "nf4"], default="none")
    for flag, default in [
        ("batch-size", 1),
        ("requests", 8),
        ("input-tokens", 64),
        ("output-tokens", 32),
        ("repeats", 3),
        ("warmup", 1),
        ("threads", 4),
    ]:
        run.add_argument("--" + flag, type=positive, default=default)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--nvml-index",
        type=int,
        default=0,
        help="Physical NVIDIA device index; match CUDA_VISIBLE_DEVICES",
    )
    run.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    run.add_argument("--out", required=True)
    report = commands.add_parser("report", help="Generate Markdown and CSV from comparable results")
    report.add_argument("files", nargs="+")
    report.add_argument("--out", default="results/report")
    report.add_argument("--readme", help="Replace text inside benchmark markers")
    suite = commands.add_parser("suite", help="Run a JSON experiment matrix in separate processes")
    suite.add_argument("config")
    suite.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not 0 < args.gpu_memory_utilization < 1:
            parser.error("--gpu-memory-utilization must be between 0 and 1")
        if args.requests % args.batch_size:
            parser.error("--requests must be divisible by --batch-size for comparable full batches")
        from .runner import run

        run(args)
    elif args.command == "report":
        from .report import report

        report(args)
    else:
        from .suite import suite

        suite(args)
