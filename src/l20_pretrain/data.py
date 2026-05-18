from __future__ import annotations

from collections.abc import Iterable, Iterator
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch.utils.data import IterableDataset

from .config import DatasetConfig


def _get_text(example: Any, text_column: str) -> str | None:
    if isinstance(example, str):
        return example
    if isinstance(example, dict):
        value = example.get(text_column)
        return value if isinstance(value, str) else None
    return None


def iter_local_text(path: str | Path) -> Iterator[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    for part in text.split("\n\n"):
        part = part.strip()
        if part:
            yield {"text": part}


def iter_filtered_texts(
    source: Iterable[Any],
    *,
    text_column: str,
    min_chars: int = 0,
    max_chars: int | None = None,
    min_score: float | None = None,
    min_int_score: int | None = None,
    max_docs: int | None = None,
) -> Iterator[str]:
    emitted = 0
    for example in source:
        text = _get_text(example, text_column)
        if not text:
            continue
        if len(text) < min_chars:
            continue
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        if isinstance(example, dict):
            if min_score is not None:
                score = example.get("score")
                if score is not None and float(score) < min_score:
                    continue
            if min_int_score is not None:
                int_score = example.get("int_score")
                if int_score is not None and int(int_score) < min_int_score:
                    continue
        yield text
        emitted += 1
        if max_docs is not None and emitted >= max_docs:
            return


class PackedTokenDataset(IterableDataset):
    def __init__(
        self,
        documents: Iterable[str],
        tokenizer: Any,
        *,
        block_size: int,
        append_eos: bool = True,
    ) -> None:
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.append_eos = append_eos

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if append_eos and eos_token_id is None:
            raise ValueError("append_eos=true requires tokenizer.eos_token_id")
        self.eos_token_id = eos_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        for text in self.documents:
            ids = tokenize_without_specials(self.tokenizer, text)
            if not ids:
                continue
            buffer.extend(ids)
            if self.append_eos:
                buffer.append(int(self.eos_token_id))
            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                del buffer[: self.block_size]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}


def collate_batch(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows], dim=0),
        "labels": torch.stack([row["labels"] for row in rows], dim=0),
    }


def tokenize_without_specials(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    except TypeError:
        return tokenizer(text, add_special_tokens=False)["input_ids"]


def create_source(config: DatasetConfig) -> Iterable[Any]:
    if config.local_text_path:
        return iter_local_text(config.local_text_path)

    from datasets import load_dataset

    direct_files = fineweb_edu_sample_files(config)
    if direct_files:
        dataset = load_dataset(
            "parquet",
            data_files=direct_files,
            split=config.split,
            streaming=config.streaming,
        )
        if config.streaming and config.shuffle_buffer > 0:
            dataset = dataset.shuffle(buffer_size=config.shuffle_buffer, seed=0)
        return dataset

    kwargs: dict[str, Any] = {
        "path": config.name,
        "split": config.split,
        "streaming": config.streaming,
    }
    if config.config_name:
        kwargs["name"] = config.config_name
    dataset = load_dataset(**kwargs)
    if config.streaming and config.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer, seed=0)
    return dataset


def fineweb_edu_sample_files(config: DatasetConfig) -> list[str] | None:
    if config.name != "HuggingFaceFW/fineweb-edu" or not config.config_name:
        return None
    match = re.fullmatch(r"sample-(\d+)BT", config.config_name)
    if not match:
        return None

    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    sample_path = f"sample/{match.group(1)}BT"
    api = HfApi(endpoint=endpoint)
    files = [
        item.path
        for item in api.list_repo_tree(
            config.name,
            repo_type="dataset",
            path_in_repo=sample_path,
            recursive=True,
            expand=False,
        )
        if item.path.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError(f"No parquet files found for {config.name}/{config.config_name}")
    return [f"{endpoint}/datasets/{config.name}/resolve/main/{path}" for path in sorted(files)]


def create_packed_dataset(
    config: DatasetConfig,
    tokenizer: Any,
    *,
    block_size: int,
) -> PackedTokenDataset:
    source = create_source(config)
    texts = iter_filtered_texts(
        source,
        text_column=config.text_column,
        min_chars=config.min_chars,
        max_chars=config.max_chars,
        min_score=config.min_score,
        min_int_score=config.min_int_score,
        max_docs=config.max_docs,
    )
    return PackedTokenDataset(
        texts,
        tokenizer,
        block_size=block_size,
        append_eos=config.append_eos,
    )
