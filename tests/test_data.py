from l20_pretrain.data import PackedTokenDataset, iter_filtered_texts


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(ch) % 31 + 1 for ch in text]}


def test_filtering_by_score_and_length() -> None:
    rows = [
        {"text": "short", "score": 5.0, "int_score": 5},
        {"text": "useful educational text", "score": 2.0, "int_score": 5},
        {"text": "high quality educational text", "score": 4.0, "int_score": 4},
    ]
    texts = list(
        iter_filtered_texts(
            rows,
            text_column="text",
            min_chars=10,
            min_score=3.0,
            min_int_score=3,
        )
    )
    assert texts == ["high quality educational text"]


def test_packing_exact_blocks() -> None:
    dataset = PackedTokenDataset(["abcd", "efgh"], TinyTokenizer(), block_size=5, append_eos=True)
    rows = list(dataset)
    assert len(rows) == 2
    assert rows[0]["input_ids"].shape[0] == 5
    assert rows[1]["labels"].shape[0] == 5
