from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from l20_pretrain.smoke_shards import create_synthetic_smoke_shards  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val shards for execution smoke tests only."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--train-blocks", type=int, default=64)
    parser.add_argument("--val-blocks", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    metadata = create_synthetic_smoke_shards(
        args.output_dir,
        block_size=args.block_size,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
