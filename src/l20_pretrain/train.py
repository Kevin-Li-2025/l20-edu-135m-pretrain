from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .env import set_default_hf_home

set_default_hf_home()

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PretrainConfig, load_config, save_config
from .data import collate_batch, create_packed_dataset
from .modeling import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Llama-style LM.")
    parser.add_argument("config", type=str, help="Path to a YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def make_optimizer(model: torch.nn.Module, config: PretrainConfig, device: torch.device) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "embed_tokens" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.trainer.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {
        "lr": config.trainer.learning_rate,
        "betas": (config.trainer.beta1, config.trainer.beta2),
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def make_scheduler(optimizer: torch.optim.Optimizer, config: PretrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, config.trainer.warmup_steps)
    total = max(warmup + 1, config.trainer.max_steps)
    min_ratio = config.trainer.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(1.0, float(step - warmup) / float(total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loader(config: PretrainConfig, tokenizer: Any) -> DataLoader:
    dataset = create_packed_dataset(
        config.dataset,
        tokenizer,
        block_size=config.model.block_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.trainer.micro_batch_size,
        collate_fn=collate_batch,
        num_workers=config.trainer.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: PretrainConfig,
    step: int,
) -> Path:
    checkpoint_dir = Path(config.output_dir) / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    save_config(config, checkpoint_dir / "pretrain_config.yaml")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


def prune_checkpoints(output_dir: str | Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    output_dir = Path(output_dir)
    checkpoints = sorted(
        path for path in output_dir.glob("step-*") if path.is_dir() and not path.is_symlink()
    )
    for checkpoint in checkpoints[:-keep_last]:
        shutil.rmtree(checkpoint)


def update_checkpoint_pointer(output_dir: str | Path, checkpoint_dir: Path, name: str = "final") -> None:
    output_dir = Path(output_dir)
    pointer = output_dir / name
    if pointer.is_symlink() or pointer.is_file():
        pointer.unlink()
    elif pointer.exists():
        (output_dir / f"{name}_checkpoint.txt").write_text(str(checkpoint_dir), encoding="utf-8")
        return

    try:
        pointer.symlink_to(checkpoint_dir.name, target_is_directory=True)
    except OSError:
        if pointer.exists():
            return
        shutil.copytree(checkpoint_dir, pointer)


def load_tokenizer(config: PretrainConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_or_create_model(config: PretrainConfig, tokenizer: Any, resume: str | None, dtype: torch.dtype) -> torch.nn.Module:
    if resume:
        return AutoModelForCausalLM.from_pretrained(resume, torch_dtype=dtype)
    return build_model(config.model, tokenizer)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    config: PretrainConfig,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    loader = build_loader(config, tokenizer)
    iterator = iter(loader)
    losses: list[float] = []
    for _ in range(config.trainer.eval_batches):
        try:
            batch = move_batch(next(iterator), device)
        except StopIteration:
            break
        with autocast_context(device, dtype):
            loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "pretrain_config.yaml")

    device = get_device(args.device)
    dtype = get_dtype(config.trainer.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = load_tokenizer(config)
    model = load_or_create_model(config, tokenizer, args.resume, dtype)
    if config.trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    if config.trainer.compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    optimizer = make_optimizer(model, config, device)
    scheduler = make_scheduler(optimizer, config)
    start_step = 0
    if args.resume:
        state_path = Path(args.resume) / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = int(state["step"])

    loader = build_loader(config, tokenizer)
    iterator = iter(loader)
    model.train()

    print(
        json.dumps(
            {
                "event": "start",
                "run_name": config.run_name,
                "device": str(device),
                "dtype": config.trainer.dtype,
                "parameters": count_parameters(unwrap_model(model)),
                "tokens_per_step": config.tokens_per_step,
                "planned_tokens": config.planned_tokens,
                "start_step": start_step,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    last_log = time.time()
    last_log_step = start_step
    for step in range(start_step + 1, config.trainer.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.trainer.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            with autocast_context(device, dtype):
                loss = model(**batch).loss / config.trainer.gradient_accumulation_steps
            total_loss += float(loss.detach().cpu()) * config.trainer.gradient_accumulation_steps
            loss.backward()

        if config.trainer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.trainer.log_interval == 0 or step == 1:
            now = time.time()
            elapsed = max(now - last_log, 1e-9)
            steps_since_log = max(1, step - last_log_step)
            last_log = now
            last_log_step = step
            tokens_per_log = config.tokens_per_step * steps_since_log
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "loss": total_loss / config.trainer.gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "tokens": step * config.tokens_per_step,
                        "tokens_per_sec_window": tokens_per_log / elapsed,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

        if config.trainer.eval_interval > 0 and step % config.trainer.eval_interval == 0:
            eval_loss = evaluate(model, config, tokenizer, device, dtype)
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "step": step,
                        "loss": eval_loss,
                        "perplexity": math.exp(eval_loss) if eval_loss < 20 else float("inf"),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

        if config.trainer.save_interval > 0 and step % config.trainer.save_interval == 0:
            checkpoint_dir = save_checkpoint(model, tokenizer, optimizer, scheduler, config, step)
            update_checkpoint_pointer(config.output_dir, checkpoint_dir)
            prune_checkpoints(config.output_dir, config.trainer.keep_last_checkpoints)
            print(
                json.dumps(
                    {"event": "checkpoint", "step": step, "path": str(checkpoint_dir)},
                    ensure_ascii=True,
                ),
                flush=True,
            )

    checkpoint_dir = save_checkpoint(model, tokenizer, optimizer, scheduler, config, config.trainer.max_steps)
    update_checkpoint_pointer(config.output_dir, checkpoint_dir)
    prune_checkpoints(config.output_dir, config.trainer.keep_last_checkpoints)
    print(
        json.dumps({"event": "done", "checkpoint": str(checkpoint_dir)}, ensure_ascii=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
