from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

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
        start_block_offset: int = 0,
    ) -> None:
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.append_eos = append_eos
        if start_block_offset < 0:
            raise ValueError("start_block_offset must be non-negative")
        self.start_block_offset = start_block_offset

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if append_eos and eos_token_id is None:
            raise ValueError("append_eos=true requires tokenizer.eos_token_id")
        self.eos_token_id = eos_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        blocks_seen = 0
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
                if blocks_seen < self.start_block_offset:
                    blocks_seen += 1
                    continue
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}


class TokenizedBlockDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        block_size: int,
        seed: int = 0,
        start_block_offset: int = 0,
        require_manifest: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(path)
        self.split = split
        self.block_size = block_size
        self.seed = seed
        if start_block_offset < 0:
            raise ValueError("start_block_offset must be non-negative")
        self.start_block_offset = start_block_offset
        self.bin_path = self.root / f"{split}.bin"
        if not self.bin_path.exists():
            detail = ""
            if split == "val":
                detail = " (create an independent val.bin; train.bin fallback is disabled)"
            raise FileNotFoundError(f"Tokenized shard not found: {self.bin_path}{detail}")
        if split == "val":
            train_path = self.root / "train.bin"
            if train_path.exists() and os.path.samefile(self.bin_path, train_path):
                raise RuntimeError(
                    "formal validation requires an independent val.bin; "
                    "val.bin resolves to the same file as train.bin"
                )

        self.metadata = self._load_metadata()
        artifact = (self.metadata.get("artifacts") or {}).get(self.bin_path.name)
        if require_manifest and not artifact:
            raise RuntimeError(
                f"immutable shard manifest is required but missing for {self.bin_path.name}"
            )
        if artifact:
            expected_bytes = int(artifact.get("bytes", -1))
            actual_bytes = self.bin_path.stat().st_size
            if expected_bytes != actual_bytes:
                raise RuntimeError(
                    f"shard size does not match manifest for {self.bin_path}: "
                    f"expected={expected_bytes}, actual={actual_bytes}"
                )
        dtype_name = self.metadata.get("dtype", "uint32")
        if dtype_name != "uint32":
            raise ValueError(f"Unsupported tokenized dtype: {dtype_name}")
        self.dtype = np.uint32
        self.num_tokens = self.bin_path.stat().st_size // np.dtype(self.dtype).itemsize
        self.num_blocks = self.num_tokens // self.block_size
        if self.num_blocks <= 0:
            raise ValueError(
                f"{self.bin_path} has {self.num_tokens} tokens, fewer than block_size={self.block_size}"
            )

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            return {}
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(self.seed + worker_id * 1_000_003)
        data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        block_ids = np.arange(worker_id, self.num_blocks, num_workers, dtype=np.int64)
        if len(block_ids) == 0:
            return

        remaining_offset = self.start_block_offset
        while True:
            order = block_ids.copy()
            rng.shuffle(order)
            if remaining_offset >= len(order):
                remaining_offset -= len(order)
                continue
            epoch_start = remaining_offset
            remaining_offset = 0
            for block_id in order[epoch_start:]:
                start = int(block_id) * self.block_size
                block = np.asarray(data[start : start + self.block_size], dtype=np.int64)
                input_ids = torch.from_numpy(block.copy())
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
    if config.revision:
        kwargs["revision"] = config.revision
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
    revision = config.revision or "main"
    sample_path = f"sample/{match.group(1)}BT"
    api = HfApi(endpoint=endpoint)
    files = [
        item.path
        for item in api.list_repo_tree(
            config.name,
            repo_type="dataset",
            revision=revision,
            path_in_repo=sample_path,
            recursive=True,
            expand=False,
        )
        if item.path.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError(f"No parquet files found for {config.name}/{config.config_name}")
    encoded_revision = quote(revision, safe="")
    return [
        f"{endpoint}/datasets/{config.name}/resolve/{encoded_revision}/{path}"
        for path in sorted(files)
    ]


def create_packed_dataset(
    config: DatasetConfig,
    tokenizer: Any,
    *,
    block_size: int,
    seed: int = 0,
    start_block_offset: int = 0,
) -> IterableDataset:
    if config.tokenized_path:
        return TokenizedBlockDataset(
            config.tokenized_path,
            split=config.split,
            block_size=block_size,
            seed=seed,
            start_block_offset=start_block_offset,
            require_manifest=config.require_manifest,
        )

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
        start_block_offset=start_block_offset,
    )
