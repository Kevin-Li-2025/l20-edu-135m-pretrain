from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from l20_pretrain.config import load_config
from l20_pretrain.train import (
    autocast_context,
    compile_model_if_requested,
    get_dtype,
    load_or_create_model,
    load_tokenizer,
    make_optimizer,
    maybe_apply_liger_kernel,
)


@dataclass(frozen=True)
class Variant:
    label: str
    micro_batch_size: int
    gradient_checkpointing: bool = False
    compile: bool = False
    liger_kernel: bool = False
    attn_implementation: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-GPU long-context training variants.")
    parser.add_argument("config")
    parser.add_argument(
        "--variants",
        default="base:2,base:3,base:4,ckpt:5,ckpt:6,compile:2,liger:2,liger:3,liger+compile:2",
        help=(
            "Comma-separated variants. Modes can include ckpt, compile, liger, and flash, "
            "e.g. flash+liger:4."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--single-variant", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_variant(raw: str) -> Variant:
    mode, micro_batch = raw.split(":", 1)
    flags = {flag for flag in mode.strip().lower().split("+") if flag and flag != "base"}
    unknown = flags - {"ckpt", "compile", "liger", "flash", "sdpa"}
    if unknown:
        raise ValueError(f"Unknown benchmark flags in {raw!r}: {sorted(unknown)}")
    if "flash" in flags and "sdpa" in flags:
        raise ValueError(f"Variant cannot request both flash and sdpa attention: {raw!r}")
    return Variant(
        label=raw,
        micro_batch_size=int(micro_batch),
        gradient_checkpointing="ckpt" in flags,
        compile="compile" in flags,
        liger_kernel="liger" in flags,
        attn_implementation="flash_attention_2" if "flash" in flags else ("sdpa" if "sdpa" in flags else None),
    )


def parse_variants(raw: str) -> list[Variant]:
    return [parse_variant(part.strip()) for part in raw.split(",") if part.strip()]


def run_training_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    grad_accumulation_steps: int,
) -> float:
    total_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accumulation_steps):
            with autocast_context(device, dtype):
                loss = model(input_ids=input_ids, labels=labels).loss / grad_accumulation_steps
            step_loss += float(loss.detach().cpu()) * grad_accumulation_steps
            loss.backward()
        optimizer.step()
        total_loss += step_loss
    return total_loss / max(1, steps)


def benchmark_variant(
    config_path: str,
    *,
    variant: Variant,
    warmup_steps: int,
    measure_steps: int,
    grad_accumulation_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg.trainer.micro_batch_size = variant.micro_batch_size
    cfg.trainer.gradient_checkpointing = variant.gradient_checkpointing
    cfg.trainer.gradient_accumulation_steps = grad_accumulation_steps
    cfg.trainer.compile = variant.compile
    cfg.trainer.liger_kernel = variant.liger_kernel
    if variant.attn_implementation:
        cfg.model.attn_implementation = variant.attn_implementation
    dtype = get_dtype(cfg.trainer.dtype)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    liger_applied = maybe_apply_liger_kernel(cfg)
    tokenizer = load_tokenizer(cfg)
    model = load_or_create_model(cfg, tokenizer, None, dtype)
    if variant.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.to(device)
    model = compile_model_if_requested(model, cfg, device)
    model.train()
    optimizer = make_optimizer(model, cfg, device)

    input_ids = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(variant.micro_batch_size, cfg.model.block_size),
        device=device,
    )
    labels = input_ids.clone()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    warmup_loss = run_training_steps(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
        device=device,
        dtype=dtype,
        steps=warmup_steps,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    measured_loss = run_training_steps(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
        device=device,
        dtype=dtype,
        steps=measure_steps,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    end.record()
    torch.cuda.synchronize()

    elapsed_sec = start.elapsed_time(end) / 1000.0
    tokens = measure_steps * grad_accumulation_steps * variant.micro_batch_size * cfg.model.block_size
    result = {
        "event": "benchmark_result",
        "variant": variant.label,
        "gradient_checkpointing": variant.gradient_checkpointing,
        "compile": variant.compile,
        "liger_kernel": variant.liger_kernel,
        "liger_applied": liger_applied,
        "attn_implementation": cfg.model.attn_implementation,
        "micro_batch_size": variant.micro_batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "measure_steps": measure_steps,
        "elapsed_sec": elapsed_sec,
        "tokens": tokens,
        "tokens_per_sec": tokens / elapsed_sec,
        "step_time_sec": elapsed_sec / measure_steps,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "warmup_loss": float(warmup_loss),
        "loss": float(measured_loss),
    }

    del optimizer, model, input_ids, labels, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_single(args: argparse.Namespace) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    variant = parse_variant(args.single_variant)
    print(json.dumps({"event": "benchmark_start", "variant": variant.label}), flush=True)
    try:
        result = benchmark_variant(
            args.config,
            variant=variant,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            grad_accumulation_steps=args.grad_accumulation_steps,
            device=device,
        )
        print(json.dumps(result), flush=True)
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "benchmark_oom",
                    "variant": variant.label,
                    "micro_batch_size": variant.micro_batch_size,
                    "error": str(exc),
                }
            ),
            flush=True,
        )
        return 0


def run_orchestrator(args: argparse.Namespace) -> int:
    variants = parse_variants(args.variants)
    output_handle = Path(args.output_jsonl).open("w", encoding="utf-8") if args.output_jsonl else None
    try:
        for variant in variants:
            command = [
                sys.executable,
                __file__,
                args.config,
                "--device",
                args.device,
                "--warmup-steps",
                str(args.warmup_steps),
                "--measure-steps",
                str(args.measure_steps),
                "--grad-accumulation-steps",
                str(args.grad_accumulation_steps),
                "--single-variant",
                variant.label,
            ]
            process = subprocess.run(command, check=False, text=True, capture_output=True)
            if process.stderr:
                sys.stderr.write(process.stderr)
                sys.stderr.flush()
            for line in process.stdout.splitlines():
                print(line, flush=True)
                if output_handle and line.startswith("{"):
                    output_handle.write(line + "\n")
                    output_handle.flush()
            if process.returncode != 0:
                return process.returncode
    finally:
        if output_handle:
            output_handle.close()
    return 0


def main() -> None:
    args = parse_args()
    if args.single_variant:
        raise SystemExit(run_single(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
