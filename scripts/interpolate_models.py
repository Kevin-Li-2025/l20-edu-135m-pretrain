#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linearly interpolate two compatible Hugging Face causal LMs."
    )
    parser.add_argument("--base", required=True, help="Base model path or Hub ID.")
    parser.add_argument("--tuned", required=True, help="Tuned model path or Hub ID.")
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    tuned = AutoModelForCausalLM.from_pretrained(
        args.tuned,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    base_state = base.state_dict()
    tuned_state = tuned.state_dict()
    if base_state.keys() != tuned_state.keys():
        missing = sorted(tuned_state.keys() - base_state.keys())
        extra = sorted(base_state.keys() - tuned_state.keys())
        raise ValueError(f"State dict keys differ; missing={missing}, extra={extra}")
    for name, base_value in base_state.items():
        tuned_value = tuned_state[name]
        if base_value.shape != tuned_value.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {tuple(base_value.shape)} != "
                f"{tuple(tuned_value.shape)}"
            )

    tuned_parameters = dict(tuned.named_parameters())
    tuned_buffers = dict(tuned.named_buffers())
    with torch.no_grad():
        for name, base_parameter in base.named_parameters():
            base_parameter.lerp_(tuned_parameters[name], args.alpha)
        for name, base_buffer in base.named_buffers():
            tuned_buffer = tuned_buffers[name]
            if base_buffer.is_floating_point():
                base_buffer.lerp_(tuned_buffer, args.alpha)
            elif not torch.equal(base_buffer, tuned_buffer):
                raise ValueError(f"Non-floating buffer differs: {name}")

    # Keep the tuned checkpoint's chat-generation contract while restoring KV
    # caching for inference. Training checkpoints intentionally disable cache.
    base.generation_config = tuned.generation_config
    base.config.use_cache = True
    base.generation_config.use_cache = True

    args.output.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tuned)
    tokenizer.save_pretrained(args.output)
    print(
        f"Saved interpolation base={args.base} tuned={args.tuned} "
        f"alpha={args.alpha} output={args.output}"
    )


if __name__ == "__main__":
    main()
