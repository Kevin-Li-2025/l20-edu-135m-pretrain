#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Print token budget for a config.")
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"run_name: {config.run_name}")
    print(f"block_size: {config.model.block_size:,}")
    print(f"micro_batch_size: {config.trainer.micro_batch_size:,}")
    print(f"gradient_accumulation_steps: {config.trainer.gradient_accumulation_steps:,}")
    print(f"tokens_per_step: {config.tokens_per_step:,}")
    print(f"max_steps: {config.trainer.max_steps:,}")
    print(f"planned_tokens: {config.planned_tokens:,}")


if __name__ == "__main__":
    main()
