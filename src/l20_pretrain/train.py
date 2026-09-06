from __future__ import annotations

import argparse
from dataclasses import replace
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import DatasetConfig, PretrainConfig, load_config, save_config
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


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - state.keys())
    if missing:
        raise RuntimeError(f"checkpoint RNG state is incomplete; missing: {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        if "torch_cuda" not in state:
            raise RuntimeError("CUDA resume requires saved torch_cuda RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def restore_trainer_state(
    state_path: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: PretrainConfig,
) -> tuple[int, int]:
    state_path = Path(state_path)
    if not state_path.exists():
        raise FileNotFoundError(
            f"exact resume requires trainer state, but it is missing: {state_path}"
        )
    if not config.dataset.tokenized_path:
        raise RuntimeError(
            "exact resume from streaming or raw text is unsupported; "
            "prepare immutable tokenized shards and resume from those"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    required = {"optimizer", "scheduler", "step", "rng_state", "consumed_train_blocks"}
    missing = sorted(required - state.keys())
    if missing:
        raise RuntimeError(
            "legacy checkpoint cannot be resumed exactly; missing trainer state: "
            + ", ".join(missing)
        )
    start_step = int(state["step"])
    consumed_train_blocks = int(state["consumed_train_blocks"])
    expected_blocks = (
        start_step
        * config.trainer.gradient_accumulation_steps
        * config.trainer.micro_batch_size
    )
    if consumed_train_blocks != expected_blocks:
        raise RuntimeError(
            "resume configuration changes the consumed-block schedule: "
            f"checkpoint={consumed_train_blocks}, config_expected={expected_blocks}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng_state"])
    return start_step, consumed_train_blocks


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
    if device.type == "cuda" and not config.trainer.deterministic:
        kwargs["fused"] = True
    elif device.type == "cuda":
        kwargs["foreach"] = False
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def learning_rate_multiplier(step: int, config: PretrainConfig) -> float:
    warmup = max(1, config.trainer.warmup_steps)
    total = max(warmup + 1, config.trainer.max_steps)
    min_ratio = config.trainer.min_lr_ratio
    if step < warmup:
        return float(step + 1) / float(warmup)
    if config.trainer.lr_schedule == "wsd":
        decay_steps = max(1, int(math.ceil(total * config.trainer.decay_fraction)))
        decay_start = max(warmup, total - decay_steps)
        if step <= decay_start:
            return 1.0
        progress = min(1.0, float(step - decay_start) / float(total - decay_start))
        return 1.0 - (1.0 - min_ratio) * progress
    progress = min(1.0, float(step - warmup) / float(total - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def make_scheduler(optimizer: torch.optim.Optimizer, config: PretrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(step, config),
    )


def build_loader(
    config: PretrainConfig,
    tokenizer: Any,
    *,
    dataset_config: DatasetConfig | None = None,
    start_block_offset: int = 0,
) -> DataLoader:
    dataset_config = dataset_config or config.dataset
    dataset = create_packed_dataset(
        dataset_config,
        tokenizer,
        block_size=config.model.block_size,
        seed=config.seed,
        start_block_offset=start_block_offset,
    )
    if start_block_offset and config.trainer.num_workers > 0:
        raise ValueError("Exact data resume requires trainer.num_workers=0")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.trainer.micro_batch_size,
        collate_fn=collate_batch,
        num_workers=config.trainer.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )


def evaluation_dataset_config(config: PretrainConfig) -> DatasetConfig:
    if config.eval_dataset is not None:
        return config.eval_dataset
    if config.dataset.tokenized_path:
        return replace(config.dataset, split="val")
    raise RuntimeError(
        "formal validation requires an explicit eval_dataset for streaming or text data"
    )


def preflight_validation_data(config: PretrainConfig, tokenizer: Any) -> None:
    """Resolve validation data before model allocation or optimizer setup."""

    if config.trainer.eval_interval <= 0:
        return
    create_packed_dataset(
        evaluation_dataset_config(config),
        tokenizer,
        block_size=config.model.block_size,
    )


def preflight_training_data(config: PretrainConfig, tokenizer: Any) -> None:
    """Reject accidental epoch rollover before allocating the model."""

    if config.dataset.allow_repetition:
        return
    dataset = create_packed_dataset(
        config.dataset,
        tokenizer,
        block_size=config.model.block_size,
        seed=config.seed,
    )
    available_blocks = getattr(dataset, "num_blocks", None)
    if available_blocks is None:
        raise RuntimeError("allow_repetition=false requires finite tokenized shards")
    required_blocks = (
        config.trainer.max_steps
        * config.trainer.gradient_accumulation_steps
        * config.trainer.micro_batch_size
    )
    if required_blocks > available_blocks:
        raise RuntimeError(
            "training plan would repeat tokenized data: "
            f"required_blocks={required_blocks}, available_blocks={available_blocks}"
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
    consumed_train_blocks: int,
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
            "rng_state": capture_rng_state(),
            "consumed_train_blocks": consumed_train_blocks,
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        revision=config.tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def maybe_apply_liger_kernel(config: PretrainConfig) -> bool:
    if not config.trainer.liger_kernel:
        return False
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
    except ImportError as exc:
        raise RuntimeError(
            "trainer.liger_kernel=true requires `liger-kernel`. "
            "Install it with `pip install 'l20-pretrain[speed]'` or `pip install liger-kernel`."
        ) from exc

    apply_liger_kernel_to_llama()
    return True


def compile_model_if_requested(
    model: torch.nn.Module,
    config: PretrainConfig,
    device: torch.device,
) -> torch.nn.Module:
    if not config.trainer.compile or not hasattr(torch, "compile") or device.type != "cuda":
        return model

    kwargs: dict[str, Any] = {}
    if config.trainer.compile_mode:
        kwargs["mode"] = config.trainer.compile_mode
    if config.trainer.compile_fullgraph is not None:
        kwargs["fullgraph"] = config.trainer.compile_fullgraph
    return torch.compile(model, **kwargs)


def configure_pretrained_model_config(config: PretrainConfig, model_name_or_path: str) -> Any:
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    original_max_position_embeddings = getattr(model_config, "max_position_embeddings", 0)
    if config.model.rope_scaling:
        rope_scaling = dict(config.model.rope_scaling)
        rope_scaling.setdefault("factor", config.model.block_size / max(1, original_max_position_embeddings))
        if rope_scaling.get("rope_type") in {"yarn", "longrope", "llama3", "dynamic"}:
            rope_scaling.setdefault("original_max_position_embeddings", original_max_position_embeddings)
        model_config.rope_scaling = rope_scaling
    if original_max_position_embeddings < config.model.block_size:
        model_config.max_position_embeddings = config.model.block_size
    if hasattr(model_config, "rope_theta"):
        model_config.rope_theta = config.model.rope_theta
    if config.model.attn_implementation:
        model_config._attn_implementation = config.model.attn_implementation
    return model_config


def load_or_create_model(config: PretrainConfig, tokenizer: Any, resume: str | None, dtype: torch.dtype) -> torch.nn.Module:
    load_kwargs: dict[str, Any] = {}
    if config.model.attn_implementation:
        load_kwargs["attn_implementation"] = config.model.attn_implementation
    if resume:
        # A resume must preserve the checkpoint's parameter dtype. Casting an
        # FP32 checkpoint to the BF16 autocast dtype changes the next update and
        # silently breaks exact continuation.
        load_kwargs["config"] = configure_pretrained_model_config(config, resume)
        return AutoModelForCausalLM.from_pretrained(resume, **load_kwargs)
    if config.init_model_name_or_path:
        load_kwargs["dtype"] = dtype
        load_kwargs["config"] = configure_pretrained_model_config(config, config.init_model_name_or_path)
        return AutoModelForCausalLM.from_pretrained(config.init_model_name_or_path, **load_kwargs)
    return build_model(config.model, tokenizer)


def estimate_flops_per_token(config: PretrainConfig, parameter_count: int) -> int:
    head_dim = config.model.hidden_size // config.model.num_attention_heads
    attention_flops = (
        12
        * config.model.num_hidden_layers
        * config.model.num_attention_heads
        * head_dim
        * config.model.block_size
    )
    return int(6 * parameter_count + attention_flops)


def estimate_mfu(
    *,
    tokens_per_sec: float,
    flops_per_token: int,
    peak_tflops: float | None,
) -> float | None:
    if not peak_tflops or peak_tflops <= 0:
        return None
    achieved_tflops = tokens_per_sec * flops_per_token / 1e12
    return achieved_tflops / peak_tflops


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    config: PretrainConfig,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    loader = build_loader(
        config,
        tokenizer,
        dataset_config=evaluation_dataset_config(config),
    )
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
        if config.trainer.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        else:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    liger_applied = maybe_apply_liger_kernel(config)
    tokenizer = load_tokenizer(config)
    preflight_training_data(config, tokenizer)
    preflight_validation_data(config, tokenizer)
    model = load_or_create_model(config, tokenizer, args.resume, dtype)
    if getattr(model.config, "max_position_embeddings", 0) < config.model.block_size:
        model.config.max_position_embeddings = config.model.block_size
    if config.trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    model = compile_model_if_requested(model, config, device)

    optimizer = make_optimizer(model, config, device)
    scheduler = make_scheduler(optimizer, config)
    start_step = 0
    consumed_train_blocks = 0
    if args.resume:
        start_step, consumed_train_blocks = restore_trainer_state(
            Path(args.resume) / "trainer_state.pt",
            optimizer,
            scheduler,
            config,
        )

    loader = build_loader(
        config,
        tokenizer,
        start_block_offset=consumed_train_blocks,
    )
    iterator = iter(loader)
    model.train()
    parameter_count = count_parameters(unwrap_model(model))
    flops_per_token = estimate_flops_per_token(config, parameter_count)

    print(
        json.dumps(
            {
                "event": "start",
                "run_name": config.run_name,
                "device": str(device),
                "dtype": config.trainer.dtype,
                "deterministic": config.trainer.deterministic,
                "parameters": parameter_count,
                "tokens_per_step": config.tokens_per_step,
                "planned_tokens": config.planned_tokens,
                "lr_schedule": config.trainer.lr_schedule,
                "start_step": start_step,
                "init_model_name_or_path": config.init_model_name_or_path,
                "block_size": config.model.block_size,
                "rope_scaling": getattr(unwrap_model(model).config, "rope_scaling", None),
                "flops_per_token_estimate": flops_per_token,
                "liger_kernel": liger_applied,
                "compile": config.trainer.compile,
                "compile_mode": config.trainer.compile_mode,
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
        consumed_train_blocks += (
            config.trainer.gradient_accumulation_steps * config.trainer.micro_batch_size
        )

        if step % config.trainer.log_interval == 0 or step == 1:
            now = time.time()
            elapsed = max(now - last_log, 1e-9)
            steps_since_log = max(1, step - last_log_step)
            last_log = now
            last_log_step = step
            tokens_per_log = config.tokens_per_step * steps_since_log
            tokens_per_sec = tokens_per_log / elapsed
            mfu = estimate_mfu(
                tokens_per_sec=tokens_per_sec,
                flops_per_token=flops_per_token,
                peak_tflops=config.trainer.mfu_peak_tflops,
            )
            payload: dict[str, Any] = {
                "event": "train",
                "step": step,
                "loss": total_loss / config.trainer.gradient_accumulation_steps,
                "lr": scheduler.get_last_lr()[0],
                "tokens": step * config.tokens_per_step,
                "tokens_per_sec_window": tokens_per_sec,
                "step_time_sec_window": elapsed / steps_since_log,
            }
            if mfu is not None:
                payload["mfu"] = mfu
                payload["mfu_pct"] = 100.0 * mfu
            print(
                json.dumps(payload, ensure_ascii=True),
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
            checkpoint_dir = save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                config,
                step,
                consumed_train_blocks,
            )
            update_checkpoint_pointer(config.output_dir, checkpoint_dir)
            prune_checkpoints(config.output_dir, config.trainer.keep_last_checkpoints)
            print(
                json.dumps(
                    {"event": "checkpoint", "step": step, "path": str(checkpoint_dir)},
                    ensure_ascii=True,
                ),
                flush=True,
            )

    checkpoint_dir = save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        config,
        config.trainer.max_steps,
        consumed_train_blocks,
    )
    update_checkpoint_pointer(config.output_dir, checkpoint_dir)
    prune_checkpoints(config.output_dir, config.trainer.keep_last_checkpoints)
    print(
        json.dumps({"event": "done", "checkpoint": str(checkpoint_dir)}, ensure_ascii=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
