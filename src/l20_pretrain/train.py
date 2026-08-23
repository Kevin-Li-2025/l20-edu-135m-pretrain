from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import datetime
import json
import math
import os
import random
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .env import set_default_hf_home

set_default_hf_home()

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import DatasetConfig, PretrainConfig, load_config, save_config
from .data import collate_batch, create_packed_dataset
from .modeling import build_model, count_parameters


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Llama-style LM.")
    parser.add_argument("config", type=str, help="Path to a YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override trainer.max_steps (useful for preflight runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir without editing the YAML config.",
    )
    return parser.parse_args()


def override_training_steps(config: PretrainConfig, max_steps: int) -> PretrainConfig:
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    trainer = config.trainer
    if trainer.lr_scheduler_type.lower() == "wsd":
        original_steps = max(1, trainer.max_steps)
        warmup_ratio = trainer.warmup_steps / original_steps
        decay_start = trainer.lr_decay_starting_step
        if decay_start is None:
            raise ValueError(
                "trainer.lr_decay_starting_step is required for lr_scheduler_type=wsd"
            )
        decay_ratio = decay_start / original_steps
        scaled_warmup = max(1, min(max_steps, round(max_steps * warmup_ratio)))
        scaled_decay_start = max(
            scaled_warmup,
            min(max_steps, round(max_steps * decay_ratio)),
        )
        trainer = replace(
            trainer,
            max_steps=max_steps,
            warmup_steps=scaled_warmup,
            lr_decay_starting_step=scaled_decay_start,
        )
    else:
        trainer = replace(trainer, max_steps=max_steps)
    return replace(config, trainer=trainer)


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


def initialize_distributed(device_name: str | None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=get_device(device_name),
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = os.environ.get(
        "TORCH_DISTRIBUTED_BACKEND",
        "nccl" if torch.cuda.is_available() else "gloo",
    )
    if backend == "nccl":
        if device_name and not device_name.startswith("cuda"):
            raise ValueError("NCCL torchrun training requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL torchrun requested, but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = get_device(device_name or "cpu")
    init_kwargs: dict[str, Any] = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "timeout": datetime.timedelta(minutes=30),
    }
    if backend == "nccl":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def distributed_barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


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
    scheduler_type = config.trainer.lr_scheduler_type.lower()
    if scheduler_type not in {"cosine", "wsd"}:
        raise ValueError("trainer.lr_scheduler_type must be one of: cosine, wsd")
    decay_start = config.trainer.lr_decay_starting_step
    if scheduler_type == "wsd":
        if decay_start is None:
            raise ValueError(
                "trainer.lr_decay_starting_step is required for lr_scheduler_type=wsd"
            )
        if not warmup <= decay_start < total:
            raise ValueError(
                "trainer.lr_decay_starting_step must be between warmup_steps "
                "and max_steps - 1"
            )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        if scheduler_type == "wsd":
            assert decay_start is not None
            if step < decay_start:
                return 1.0
            progress = min(1.0, float(step - decay_start) / float(total - decay_start))
            return 1.0 - (1.0 - min_ratio) * progress
        progress = min(1.0, float(step - warmup) / float(total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loader(
    config: PretrainConfig,
    tokenizer: Any,
    *,
    distributed: DistributedContext | None = None,
    split: str | None = None,
    dataset_config: DatasetConfig | None = None,
    start_block_offset: int = 0,
) -> DataLoader:
    distributed = distributed or DistributedContext(0, 0, 1, torch.device("cpu"))
    dataset_config = dataset_config or config.dataset
    if split is not None:
        dataset_config = replace(dataset_config, split=split)
    configured_offset = (
        dataset_config.start_block_offset_per_rank
        if dataset_config.split == "train"
        else 0
    )
    total_start_block_offset = configured_offset + start_block_offset
    dataset = create_packed_dataset(
        dataset_config,
        tokenizer,
        block_size=config.model.block_size,
        seed=config.seed,
        rank=distributed.rank,
        world_size=distributed.world_size,
        start_block_offset=total_start_block_offset,
    )
    if total_start_block_offset and config.trainer.num_workers > 0:
        raise ValueError(
            "Exact tokenized-data resume requires trainer.num_workers=0"
        )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.trainer.micro_batch_size,
        "collate_fn": collate_batch,
        "num_workers": config.trainer.num_workers,
        "pin_memory": distributed.device.type == "cuda",
    }
    if config.trainer.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(
        **loader_kwargs,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    seen: set[int] = set()
    while id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        if isinstance(unwrapped, DistributedDataParallel):
            unwrapped = unwrapped.module
            continue
        original = getattr(unwrapped, "_orig_mod", None)
        if original is not None:
            unwrapped = original
            continue
        break
    return unwrapped


def wrap_distributed_model(
    model: torch.nn.Module,
    context: DistributedContext,
    config: PretrainConfig,
) -> torch.nn.Module:
    bucket_cap_mb = config.trainer.ddp_bucket_cap_mb
    if bucket_cap_mb <= 0:
        raise ValueError("trainer.ddp_bucket_cap_mb must be positive")
    gradient_compression = config.trainer.ddp_gradient_compression.lower()
    if gradient_compression not in {"none", "bf16"}:
        raise ValueError(
            "trainer.ddp_gradient_compression must be one of: none, bf16"
        )
    if not context.enabled:
        return model
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
        "bucket_cap_mb": bucket_cap_mb,
        "static_graph": config.trainer.ddp_static_graph,
    }
    if context.device.type == "cuda":
        kwargs["device_ids"] = [context.local_rank]
        kwargs["output_device"] = context.local_rank
    wrapped = DistributedDataParallel(model, **kwargs)
    if gradient_compression == "bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        wrapped.register_comm_hook(
            state=None,
            hook=default_hooks.bf16_compress_hook,
        )
    return wrapped


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
    with model_for_serialization(model) as serializable_model:
        serializable_model.save_pretrained(checkpoint_dir, safe_serialization=True)
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

    profile = config.trainer.liger_profile
    if profile == "full":
        apply_liger_kernel_to_llama()
    elif profile == "fused_ce_only":
        apply_liger_kernel_to_llama(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
        )
    else:
        raise ValueError(
            "trainer.liger_profile must be one of: full, fused_ce_only"
        )
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
    scope = config.trainer.compile_scope
    if scope == "model":
        return torch.compile(model, **kwargs)
    if scope == "backbone":
        backbone = getattr(model, "model", None)
        if not isinstance(backbone, torch.nn.Module):
            raise ValueError(
                "trainer.compile_scope=backbone requires a model.model module"
            )
        model.model = torch.compile(backbone, **kwargs)
        return model
    raise ValueError("trainer.compile_scope must be one of: model, backbone")


@contextmanager
def model_for_serialization(model: torch.nn.Module) -> Any:
    """Temporarily expose original modules so HF checkpoints have canonical keys."""
    root = unwrap_model(model)
    replacements: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def unwrap_children(parent: torch.nn.Module) -> None:
        for name, child in list(parent.named_children()):
            original = getattr(child, "_orig_mod", None)
            if isinstance(original, torch.nn.Module):
                replacements.append((parent, name, child))
                setattr(parent, name, original)
                unwrap_children(original)
            else:
                unwrap_children(child)

    unwrap_children(root)
    try:
        yield root
    finally:
        for parent, name, compiled in reversed(replacements):
            setattr(parent, name, compiled)


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


def load_or_create_model(
    config: PretrainConfig,
    tokenizer: Any,
    resume: str | None,
    parameter_dtype: torch.dtype,
) -> torch.nn.Module:
    load_kwargs: dict[str, Any] = {"torch_dtype": parameter_dtype}
    if config.model.attn_implementation:
        load_kwargs["attn_implementation"] = config.model.attn_implementation
    if resume:
        load_kwargs["config"] = configure_pretrained_model_config(config, resume)
        return AutoModelForCausalLM.from_pretrained(resume, **load_kwargs)
    if config.init_model_name_or_path:
        load_kwargs["config"] = configure_pretrained_model_config(config, config.init_model_name_or_path)
        return AutoModelForCausalLM.from_pretrained(config.init_model_name_or_path, **load_kwargs)
    return build_model(config.model, tokenizer)


def load_anchor_model(
    config: PretrainConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module | None:
    weight = config.trainer.anchor_kl_weight
    if weight < 0:
        raise ValueError("trainer.anchor_kl_weight must be non-negative")
    if weight == 0:
        return None
    if not config.anchor_model_name_or_path:
        raise ValueError(
            "anchor_model_name_or_path is required when trainer.anchor_kl_weight is positive"
        )
    if config.trainer.anchor_kl_temperature <= 0:
        raise ValueError("trainer.anchor_kl_temperature must be positive")
    if config.trainer.anchor_kl_stride <= 0:
        raise ValueError("trainer.anchor_kl_stride must be positive")
    if config.trainer.anchor_kl_chunk_size <= 0:
        raise ValueError("trainer.anchor_kl_chunk_size must be positive")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "config": configure_pretrained_model_config(config, config.anchor_model_name_or_path),
    }
    if config.model.attn_implementation:
        load_kwargs["attn_implementation"] = config.model.attn_implementation
    anchor = AutoModelForCausalLM.from_pretrained(
        config.anchor_model_name_or_path,
        **load_kwargs,
    )
    anchor.requires_grad_(False)
    anchor.eval()
    anchor.config.use_cache = False
    anchor.to(device)
    return anchor


def token_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    stride: int = 1,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Return next-token KL(teacher || student) over non-masked labels.

    Sequence chunks keep the temporary float32 softmax tensors bounded for
    long-context runs. ``stride`` allows a deterministic position sample when
    a full-vocabulary teacher pass would otherwise make KL reduction expensive.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student and teacher logits differ: {student_logits.shape} != {teacher_logits.shape}"
        )
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if student_logits.shape[:2] != labels.shape:
        raise ValueError("Logit batch/sequence dimensions must match labels")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    prediction_length = student_logits.shape[1] - 1
    if prediction_length <= 0:
        raise ValueError("At least two sequence positions are required for next-token KL")

    total_kl = student_logits.new_zeros((), dtype=torch.float32)
    total_tokens = 0
    span = stride * chunk_size
    target_labels = labels[:, 1:]
    for start in range(0, prediction_length, span):
        stop = min(prediction_length, start + span)
        positions = slice(start, stop, stride)
        valid = target_labels[:, positions].ne(-100)
        valid_tokens = int(valid.sum().item())
        if valid_tokens == 0:
            continue

        student_chunk = student_logits[:, :-1, :][:, positions, :].float() / temperature
        teacher_chunk = teacher_logits[:, :-1, :][:, positions, :].float() / temperature
        teacher_probs = F.softmax(teacher_chunk, dim=-1)
        student_log_probs = F.log_softmax(student_chunk, dim=-1)
        per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)
        total_kl = total_kl + per_token.masked_select(valid).sum()
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("No supervised next-token positions are available for anchor KL")
    return total_kl * (temperature * temperature / total_tokens)


def training_stream_schedule(config: PretrainConfig) -> tuple[str, ...]:
    accumulation_steps = config.trainer.gradient_accumulation_steps
    retention_steps = config.trainer.retention_gradient_accumulation_steps
    if accumulation_steps <= 0:
        raise ValueError("trainer.gradient_accumulation_steps must be positive")
    if config.retention_dataset is None:
        if retention_steps != 0:
            raise ValueError(
                "retention_gradient_accumulation_steps requires retention_dataset"
            )
        return ("target",) * accumulation_steps
    if not 0 < retention_steps < accumulation_steps:
        raise ValueError(
            "retention_gradient_accumulation_steps must be between 1 and "
            "gradient_accumulation_steps - 1 for two-stream training"
        )
    return ("retention",) * retention_steps + (
        "target",
    ) * (accumulation_steps - retention_steps)


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
    *,
    distributed: DistributedContext | None = None,
    dataset_config: DatasetConfig | None = None,
) -> float:
    distributed = distributed or DistributedContext(0, 0, 1, device)
    model.eval()
    selected_dataset = dataset_config or config.dataset
    split = "val" if selected_dataset.tokenized_path else None
    loader = build_loader(
        config,
        tokenizer,
        distributed=distributed,
        split=split,
        dataset_config=selected_dataset,
    )
    iterator = iter(loader)
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    loss_count = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(config.trainer.eval_batches):
        try:
            batch = move_batch(next(iterator), device)
        except StopIteration:
            break
        with autocast_context(device, dtype):
            loss = model(**batch).loss
        loss_sum += loss.detach().float()
        loss_count += 1
    stats = torch.stack((loss_sum, loss_count))
    if distributed.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    if stats[1].item() == 0:
        return float("nan")
    return float((stats[0] / stats[1]).item())


def run_training(args: argparse.Namespace, distributed: DistributedContext) -> None:
    config = load_config(args.config)
    if args.max_steps is not None:
        config = override_training_steps(config, args.max_steps)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    set_seed(config.seed + distributed.rank)

    output_dir = Path(config.output_dir)
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "pretrain_config.yaml")
    distributed_barrier(distributed)

    device = distributed.device
    dtype = get_dtype(config.trainer.dtype)
    parameter_dtype = get_dtype(config.trainer.parameter_dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    liger_applied = maybe_apply_liger_kernel(config)
    tokenizer = load_tokenizer(config)
    model = load_or_create_model(config, tokenizer, args.resume, parameter_dtype)
    if getattr(model.config, "max_position_embeddings", 0) < config.model.block_size:
        model.config.max_position_embeddings = config.model.block_size
    model.config.use_cache = False
    if config.trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    anchor_model = load_anchor_model(config, dtype=dtype, device=device)
    stream_schedule = training_stream_schedule(config)

    model = compile_model_if_requested(model, config, device)
    model = wrap_distributed_model(model, distributed, config)

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

    target_micro_steps = (
        config.trainer.gradient_accumulation_steps
        - config.trainer.retention_gradient_accumulation_steps
    )
    target_loader = build_loader(
        config,
        tokenizer,
        distributed=distributed,
        start_block_offset=(
            start_step
            * target_micro_steps
            * config.trainer.micro_batch_size
        ),
    )
    target_iterator = iter(target_loader)
    retention_loader = (
        build_loader(
            config,
            tokenizer,
            distributed=distributed,
            dataset_config=config.retention_dataset,
            start_block_offset=(
                start_step
                * config.trainer.retention_gradient_accumulation_steps
                * config.trainer.micro_batch_size
            ),
        )
        if config.retention_dataset is not None
        else None
    )
    retention_iterator = iter(retention_loader) if retention_loader is not None else None
    model.train()
    parameter_count = count_parameters(unwrap_model(model))
    flops_per_token = estimate_flops_per_token(config, parameter_count)
    tokens_per_step = config.tokens_per_step * distributed.world_size
    retention_tokens_per_step = (
        config.retention_tokens_per_step * distributed.world_size
    )
    target_tokens_per_step = config.target_tokens_per_step * distributed.world_size
    peak_tflops = config.trainer.mfu_peak_tflops
    if peak_tflops is not None:
        peak_tflops *= distributed.world_size

    if distributed.is_main:
        print(
            json.dumps(
                {
                    "event": "start",
                    "run_name": config.run_name,
                    "device": str(device),
                    "distributed_backend": distributed.backend,
                    "world_size": distributed.world_size,
                    "dtype": config.trainer.dtype,
                    "parameter_dtype": config.trainer.parameter_dtype,
                    "parameters": parameter_count,
                    "tokens_per_step_per_rank": config.tokens_per_step,
                    "tokens_per_step": tokens_per_step,
                    "planned_tokens": tokens_per_step * config.trainer.max_steps,
                    "start_step": start_step,
                    "dataset_start_block_offset_per_rank": (
                        config.dataset.start_block_offset_per_rank
                    ),
                    "init_model_name_or_path": config.init_model_name_or_path,
                    "anchor_model_name_or_path": config.anchor_model_name_or_path,
                    "anchor_kl_weight": config.trainer.anchor_kl_weight,
                    "anchor_kl_temperature": config.trainer.anchor_kl_temperature,
                    "anchor_kl_stride": config.trainer.anchor_kl_stride,
                    "anchor_kl_scope": (
                        "retention" if config.retention_dataset is not None else "all"
                    ),
                    "stream_schedule": stream_schedule,
                    "retention_tokens_per_step": retention_tokens_per_step,
                    "target_tokens_per_step": target_tokens_per_step,
                    "block_size": config.model.block_size,
                    "rope_scaling": getattr(unwrap_model(model).config, "rope_scaling", None),
                    "flops_per_token_estimate": flops_per_token,
                    "mfu_peak_tflops_total": peak_tflops,
                    "liger_kernel": liger_applied,
                    "liger_profile": config.trainer.liger_profile,
                    "compile": config.trainer.compile,
                    "compile_mode": config.trainer.compile_mode,
                    "compile_scope": config.trainer.compile_scope,
                    "ddp_bucket_cap_mb": config.trainer.ddp_bucket_cap_mb,
                    "ddp_static_graph": config.trainer.ddp_static_graph,
                    "ddp_gradient_compression": config.trainer.ddp_gradient_compression,
                    "lr_scheduler_type": config.trainer.lr_scheduler_type,
                    "lr_decay_starting_step": config.trainer.lr_decay_starting_step,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    last_log = time.time()
    last_log_step = start_step
    for step in range(start_step + 1, config.trainer.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_metrics = torch.zeros(5, dtype=torch.float32, device=device)
        anchor_batches = 0
        stream_counts = {"target": 0, "retention": 0}
        for micro_step, stream in enumerate(stream_schedule):
            if stream == "retention":
                assert retention_loader is not None and retention_iterator is not None
                loader = retention_loader
                iterator = retention_iterator
            else:
                loader = target_loader
                iterator = target_iterator
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if stream == "retention":
                retention_iterator = iterator
            else:
                target_iterator = iterator
            batch = move_batch(batch, device)
            should_sync = micro_step == len(stream_schedule) - 1
            sync_context = (
                nullcontext()
                if should_sync or not distributed.enabled
                else model.no_sync()  # type: ignore[union-attr]
            )
            with sync_context:
                with autocast_context(device, dtype):
                    if (
                        config.trainer.compile
                        and device.type == "cuda"
                        and hasattr(torch.compiler, "cudagraph_mark_step_begin")
                    ):
                        torch.compiler.cudagraph_mark_step_begin()
                    outputs = model(**batch)
                    lm_loss = outputs.loss
                    anchor_kl = lm_loss.new_zeros(())
                    apply_anchor = anchor_model is not None and (
                        config.retention_dataset is None or stream == "retention"
                    )
                    if apply_anchor:
                        anchor_inputs = {
                            key: value
                            for key, value in batch.items()
                            if key in {"input_ids", "attention_mask", "position_ids"}
                        }
                        with torch.no_grad():
                            teacher_logits = anchor_model(**anchor_inputs).logits
                        anchor_kl = token_kl_divergence(
                            outputs.logits,
                            teacher_logits,
                            batch["labels"],
                            temperature=config.trainer.anchor_kl_temperature,
                            stride=config.trainer.anchor_kl_stride,
                            chunk_size=config.trainer.anchor_kl_chunk_size,
                        )
                        anchor_batches += 1
                    combined_loss = lm_loss + config.trainer.anchor_kl_weight * anchor_kl
                    loss = combined_loss / config.trainer.gradient_accumulation_steps
                step_metrics[0] += combined_loss.detach().float()
                step_metrics[1] += lm_loss.detach().float()
                step_metrics[2] += anchor_kl.detach().float()
                stream_metric_index = 3 if stream == "target" else 4
                step_metrics[stream_metric_index] += lm_loss.detach().float()
                stream_counts[stream] += 1
                loss.backward()

        grad_norm_tensor: torch.Tensor | None = None
        if config.trainer.grad_clip > 0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.trainer.grad_clip,
                error_if_nonfinite=True,
            )
        optimizer.step()
        scheduler.step()

        log_due = step == 1 or (
            config.trainer.log_interval > 0 and step % config.trainer.log_interval == 0
        )
        if log_due:
            reduced_metrics = step_metrics.clone()
            if distributed.enabled:
                dist.all_reduce(reduced_metrics, op=dist.ReduceOp.SUM)
                reduced_metrics /= distributed.world_size
            now = time.time()
            elapsed = max(now - last_log, 1e-9)
            steps_since_log = max(1, step - last_log_step)
            last_log = now
            last_log_step = step
            tokens_per_log = tokens_per_step * steps_since_log
            tokens_per_sec = tokens_per_log / elapsed
            mfu = estimate_mfu(
                tokens_per_sec=tokens_per_sec,
                flops_per_token=flops_per_token,
                peak_tflops=peak_tflops,
            )
            metric_values = reduced_metrics.cpu().tolist()
            if not all(math.isfinite(value) for value in metric_values):
                raise FloatingPointError(
                    f"Non-finite loss component at step {step}: {metric_values}"
                )
            payload: dict[str, Any] = {
                "event": "train",
                "step": step,
                "loss": metric_values[0] / config.trainer.gradient_accumulation_steps,
                "lm_loss": metric_values[1] / config.trainer.gradient_accumulation_steps,
                "lr": scheduler.get_last_lr()[0],
                "tokens": step * tokens_per_step,
                "tokens_per_sec_window": tokens_per_sec,
                "step_time_sec_window": elapsed / steps_since_log,
            }
            if grad_norm_tensor is not None:
                payload["grad_norm"] = float(grad_norm_tensor.detach().item())
            if anchor_batches:
                payload["anchor_kl"] = metric_values[2] / anchor_batches
            for stream, count in stream_counts.items():
                if count:
                    index = 3 if stream == "target" else 4
                    payload[f"{stream}_lm_loss"] = metric_values[index] / count
            if mfu is not None:
                payload["mfu"] = mfu
                payload["mfu_pct"] = 100.0 * mfu
            if distributed.is_main:
                print(
                    json.dumps(payload, ensure_ascii=True),
                    flush=True,
                )

        if config.trainer.eval_interval > 0 and step % config.trainer.eval_interval == 0:
            eval_loss = evaluate(
                model,
                config,
                tokenizer,
                device,
                dtype,
                distributed=distributed,
            )
            retention_eval_loss = (
                evaluate(
                    model,
                    config,
                    tokenizer,
                    device,
                    dtype,
                    distributed=distributed,
                    dataset_config=config.retention_dataset,
                )
                if config.retention_dataset is not None
                else None
            )
            eval_payload: dict[str, Any] = {
                "event": "eval",
                "step": step,
                "loss": eval_loss,
                "perplexity": math.exp(eval_loss) if eval_loss < 20 else float("inf"),
            }
            if retention_eval_loss is not None:
                eval_payload["retention_loss"] = retention_eval_loss
                eval_payload["retention_perplexity"] = (
                    math.exp(retention_eval_loss)
                    if retention_eval_loss < 20
                    else float("inf")
                )
            if distributed.is_main:
                print(
                    json.dumps(eval_payload, ensure_ascii=True),
                    flush=True,
                )

        if config.trainer.save_interval > 0 and step % config.trainer.save_interval == 0:
            if distributed.is_main:
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
            distributed_barrier(distributed)

    if distributed.is_main:
        checkpoint_dir = save_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            config,
            config.trainer.max_steps,
        )
        update_checkpoint_pointer(config.output_dir, checkpoint_dir)
        prune_checkpoints(config.output_dir, config.trainer.keep_last_checkpoints)
        print(
            json.dumps({"event": "done", "checkpoint": str(checkpoint_dir)}, ensure_ascii=True),
            flush=True,
        )
    distributed_barrier(distributed)


def main() -> None:
    args = parse_args()
    distributed = initialize_distributed(args.device)
    try:
        run_training(args, distributed)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
