import numpy as np
import pytest
import torch
import json

from l20_pretrain.data import (
    PackedTokenDataset,
    TokenizedBlockDataset,
    iter_filtered_texts,
)


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


def test_tokenized_validation_does_not_silently_use_training_shard(tmp_path) -> None:
    (tmp_path / "train.bin").write_bytes(np.arange(16, dtype=np.uint32).tobytes())

    with pytest.raises(FileNotFoundError, match="independent val.bin"):
        TokenizedBlockDataset(tmp_path, split="val", block_size=4)


def test_tokenized_validation_rejects_same_underlying_file(tmp_path) -> None:
    (tmp_path / "train.bin").write_bytes(np.arange(16, dtype=np.uint32).tobytes())
    (tmp_path / "val.bin").hardlink_to(tmp_path / "train.bin")

    with pytest.raises(RuntimeError, match="same file as train.bin"):
        TokenizedBlockDataset(tmp_path, split="val", block_size=4)


def test_tokenized_resume_offset_matches_uninterrupted_order(tmp_path) -> None:
    values = np.arange(48, dtype=np.uint32)
    (tmp_path / "train.bin").write_bytes(values.tobytes())

    uninterrupted = iter(TokenizedBlockDataset(tmp_path, split="train", block_size=4, seed=17))
    expected = [next(uninterrupted)["input_ids"] for _ in range(17)]
    resumed = iter(
        TokenizedBlockDataset(
            tmp_path,
            split="train",
            block_size=4,
            seed=17,
            start_block_offset=11,
        )
    )
    actual = [next(resumed)["input_ids"] for _ in range(6)]

    assert all(torch.equal(left, right) for left, right in zip(expected[11:], actual))


def test_required_manifest_is_fail_closed(tmp_path) -> None:
    (tmp_path / "train.bin").write_bytes(np.arange(16, dtype=np.uint32).tobytes())

    with pytest.raises(RuntimeError, match="manifest is required"):
        TokenizedBlockDataset(
            tmp_path,
            split="train",
            block_size=4,
            require_manifest=True,
        )


def test_manifest_size_is_checked_on_loader_startup(tmp_path) -> None:
    train = tmp_path / "train.bin"
    train.write_bytes(np.arange(16, dtype=np.uint32).tobytes())
    (tmp_path / "metadata.json").write_text(
        json.dumps({"artifacts": {"train.bin": {"bytes": train.stat().st_size + 4}}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="size does not match manifest"):
        TokenizedBlockDataset(tmp_path, split="train", block_size=4)
