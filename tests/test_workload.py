from llmbench.workload import make_workload


class Tokenizer:
    def encode(self, text, add_special_tokens):
        return list(text.encode())


def test_equal_length_reproducible_workload():
    first, digest = make_workload(Tokenizer(), 8, 128)
    second, same_digest = make_workload(Tokenizer(), 8, 128)
    assert first == second and digest == same_digest
    assert {len(prompt) for prompt in first} == {128}
    assert len({tuple(prompt) for prompt in first}) == 8
    assert make_workload(Tokenizer(), 8, 64)[1] != digest
