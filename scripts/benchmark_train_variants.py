from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
import time

import torch

from l20_pretrain.config import load_config
from l20_pretrain.train import (
    autocast_context,
    get_dtype,
    load_or_create_model,
    load_tokenizer,
    make_optimizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-GPU long-context training variants.")
    parser.add_argument("config")
    parser.add_argument(
        "--variants",
        default="ckpt:6,ckpt:8,nockpt:2,nockpt:3,nockpt:4,nockpt:5,nockpt:6",
        help="Comma-separated checkpointing:micro_batch variants.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_variants(raw: str) -> list[tuple[bool, int]]:
    variants = []
    for part in raw.split(","):
        mode, mb = part.split(":", 1)
        variants.append((mode.strip().lower() == "ckpt", int(mb)))
    return variants


def benchmark_variant(config_path: str, *, micro_batch_size: int, gradient_checkpointing: bool, device: torch.device) -> dict:
    cfg = load_config(config_path)
    cfg.trainer.micro_batch_size = micro_batch_size
    cfg.trainer.gradient_checkpointing = gradient_checkpointing
    cfg.trainer.gradient_accumulation_steps = 1
    dtype = get_dtype(cfg.trainer.dtype)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    tokenizer = load_tokenizer(cfg)
    model = load_or_create_model(cfg, tokenizer, None, dtype)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.to(device)
    model.train()
    optimizer = make_optimizer(model, cfg, device)

    input_ids = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(micro_batch_size, cfg.model.block_size),
        device=device,
    )
    labels = input_ids.clone()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, dtype):
        loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()

    elapsed_sec = start.elapsed_time(end) / 1000.0
    result = {
        "gradient_checkpointing": gradient_checkpointing,
        "micro_batch_size": micro_batch_size,
        "elapsed_sec": elapsed_sec,
        "tokens_per_sec_micro": micro_batch_size * cfg.model.block_size / elapsed_sec,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "loss": float(loss.detach().cpu()),
    }

    del optimizer, model, input_ids, labels, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    for gradient_checkpointing, micro_batch_size in parse_variants(args.variants):
        payload = {
            "event": "benchmark_start",
            "gradient_checkpointing": gradient_checkpointing,
            "micro_batch_size": micro_batch_size,
        }
        print(json.dumps(payload), flush=True)
        try:
            result = benchmark_variant(
                args.config,
                micro_batch_size=micro_batch_size,
                gradient_checkpointing=gradient_checkpointing,
                device=device,
            )
            print(json.dumps({"event": "benchmark_result", **result}), flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "event": "benchmark_oom",
                        "gradient_checkpointing": gradient_checkpointing,
                        "micro_batch_size": micro_batch_size,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
