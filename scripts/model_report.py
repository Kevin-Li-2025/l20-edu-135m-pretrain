#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.env import set_default_hf_home

set_default_hf_home()

from transformers import AutoTokenizer

from l20_pretrain.config import load_config
from l20_pretrain.modeling import build_model, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Report model shape, parameter count, and token budget.")
    parser.add_argument("configs", nargs="+", type=str)
    args = parser.parse_args()

    for path in args.configs:
        config = load_config(path)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        model = build_model(config.model, tokenizer)
        print(path)
        print(f"  params: {count_parameters(model) / 1e6:.1f}M")
        print(f"  tokenizer: {config.tokenizer_name}")
        print(f"  dataset: {config.dataset.name}/{config.dataset.config_name}")
        print(f"  shape: L{config.model.num_hidden_layers} H{config.model.hidden_size} I{config.model.intermediate_size}")
        print(f"  heads: q={config.model.num_attention_heads} kv={config.model.num_key_value_heads}")
        print(f"  tied_embeddings: {config.model.tie_word_embeddings}")
        print(f"  tokens_per_step: {config.tokens_per_step:,}")
        print(f"  planned_tokens: {config.planned_tokens:,}")


if __name__ == "__main__":
    main()
