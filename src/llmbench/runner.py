import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import HFBackend, VLLMBackend
from .metrics import MemorySampler, summarize
from .workload import make_workload


def versions():
    found = {}
    for name in ("torch", "transformers", "vllm", "bitsandbytes", "psutil"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def run(args):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    # Resolve once and use the same immutable revision for tokenizer and model.
    # Pinned SHA works with cached models even when offline.
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        args.revision = model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompts, digest = make_workload(tokenizer, args.requests, args.input_tokens)
    start = time.perf_counter()
    engine = HFBackend(args) if args.backend == "hf" else VLLMBackend(args)
    engine.sync()
    load_seconds = time.perf_counter() - start
    batches = [
        engine.prepare(prompts[i : i + args.batch_size])
        for i in range(0, len(prompts), args.batch_size)
    ]
    for _ in range(args.warmup):
        engine.generate(batches[0])
        engine.sync()
    records, sample_ids = [], None
    with MemorySampler(args.nvml_index if args.device == "cuda" else None) as memory:
        for repeat in range(args.repeats):
            for index, batch in enumerate(batches):
                engine.sync()
                start = time.perf_counter()
                raw = engine.generate(batch)
                engine.sync()
                elapsed = time.perf_counter() - start
                tokens = engine.extract(raw, batch)
                expected = min(args.batch_size, args.requests - index * args.batch_size)
                if len(tokens) != expected or any(len(t) != args.output_tokens for t in tokens):
                    raise RuntimeError("Backend did not honor fixed request/output counts")
                records.append(
                    {
                        "repeat": repeat,
                        "batch": index,
                        "seconds": elapsed,
                        "requests": len(tokens),
                        "output_tokens": sum(map(len, tokens)),
                    }
                )
                if sample_ids is None:
                    sample_ids = tokens[0]
    result = {
        "schema_version": 1,
        "status": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(args).items() if k not in ("func", "out")},
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "packages": versions(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0)
            if args.device == "cuda"
            else platform.processor(),
        },
        "workload_sha256": digest,
        "load_seconds": load_seconds,
        "metrics": summarize(records),
        "memory": memory.result(),
        "batches": records,
        "sample": {
            "output_token_ids": sample_ids,
            "text": tokenizer.decode(sample_ids, skip_special_tokens=True),
        },
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved {target}")
