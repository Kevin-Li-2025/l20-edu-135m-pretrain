from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import artifact_record


def create_synthetic_smoke_shards(
    output_dir: str | Path,
    *,
    block_size: int,
    train_blocks: int,
    val_blocks: int,
    vocab_size: int,
    seed: int,
) -> dict[str, Any]:
    if min(block_size, train_blocks, val_blocks, vocab_size) <= 0:
        raise ValueError("block_size, block counts, and vocab_size must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)
    train = train_rng.integers(
        0,
        vocab_size,
        size=block_size * train_blocks,
        dtype=np.uint32,
    )
    val = val_rng.integers(
        0,
        vocab_size,
        size=block_size * val_blocks,
        dtype=np.uint32,
    )
    train.tofile(train_path)
    val.tofile(val_path)

    metadata = {
        "schema_version": 1,
        "dtype": "uint32",
        "source": "deterministic-synthetic-smoke-only",
        "quality_evidence": False,
        "seed": seed,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "train_blocks": train_blocks,
        "val_blocks": val_blocks,
        "artifacts": {
            train_path.name: artifact_record(train_path),
            val_path.name: artifact_record(val_path),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
