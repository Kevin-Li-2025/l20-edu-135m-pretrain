from __future__ import annotations

import argparse
import math

from .env import set_default_hf_home

set_default_hf_home()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config
from .train import evaluate, get_device, get_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint perplexity.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype).to(device)
    loss = evaluate(model, config, tokenizer, device, dtype)
    perplexity = math.exp(loss) if loss < 20 else float("inf")
    print(f"loss={loss:.4f} perplexity={perplexity:.2f}")


if __name__ == "__main__":
    main()
