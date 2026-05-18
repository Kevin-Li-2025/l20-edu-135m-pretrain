from __future__ import annotations

import argparse

from .env import set_default_hf_home

set_default_hf_home()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import get_device, get_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--prompt", type=str, default="The future of education is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
