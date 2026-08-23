from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic packed tokens for throughput tests.")
    parser.add_argument("--output-dir", default="/tmp/l20_synthetic_2k")
    parser.add_argument("--tokens", type=int, default=64_000_000)
    parser.add_argument("--vocab-size", type=int, default=49_152)
    parser.add_argument("--block-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens < args.block_size:
        raise ValueError("tokens must be at least one block")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    usable_tokens = args.tokens - args.tokens % args.block_size
    rng = np.random.default_rng(args.seed)
    train_path = output_dir / "train.bin"
    mapped = np.memmap(train_path, mode="w+", dtype=np.uint32, shape=(usable_tokens,))
    chunk_size = 4_000_000
    for start in range(0, usable_tokens, chunk_size):
        stop = min(usable_tokens, start + chunk_size)
        mapped[start:stop] = rng.integers(
            0,
            args.vocab_size,
            size=stop - start,
            dtype=np.uint32,
        )
    mapped.flush()
    metadata = {
        "dtype": "uint32",
        "synthetic": True,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "block_size": args.block_size,
        "train_tokens": usable_tokens,
        "train_blocks": usable_tokens // args.block_size,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
