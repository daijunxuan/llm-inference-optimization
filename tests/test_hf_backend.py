from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmbench.backends import HFBackend  # noqa: E402


@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_real_generation_exact_budget(monkeypatch, quantization):
    """Exercise real HF generation and CPU quantization, without network/model download."""
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model
    )
    args = SimpleNamespace(
        device="cpu",
        quantization=quantization,
        dtype="float32",
        revision="test",
        model="test",
        output_tokens=4,
    )
    engine = HFBackend(args)
    inputs = engine.prepare([[1, 2, 3], [4, 5, 6]])
    result = engine.extract(engine.generate(inputs), inputs)
    assert len(result) == 2
    assert all(len(tokens) == 4 for tokens in result)
