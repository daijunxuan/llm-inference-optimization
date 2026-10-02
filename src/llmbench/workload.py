import hashlib
import json

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPTS = [
    "Explain why batching can improve the throughput of a language model.",
    "Describe the difference between latency and throughput with an example.",
    "What is the purpose of a key-value cache in transformer inference?",
    "Explain how weight quantization changes memory use and numerical precision.",
    "Write a short Python function that computes the mean of a list of numbers.",
    "Give three practical reasons to use containers for machine learning experiments.",
    "Summarize the tradeoff between larger batches and request response time.",
    "Explain why warming up a GPU matters when measuring inference speed.",
]


def make_workload(tokenizer, requests, input_tokens):
    """Identical token IDs across engines; exactly input_tokens per request.

    These are synthetic raw-text prompts, not an instruction-quality dataset.
    """
    result = []
    for i in range(requests):
        text = f"Task {i}: " + PROMPTS[i % len(PROMPTS)] + "\nAnswer:"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Tokenizer produced an empty prompt")
        result.append((ids * ((input_tokens // len(ids)) + 1))[:input_tokens])
    digest = hashlib.sha256(json.dumps(result).encode()).hexdigest()
    return result, digest
