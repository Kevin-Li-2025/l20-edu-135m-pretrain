from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from .env import set_default_hf_home

set_default_hf_home()

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import _clean_nulls
from .data import TokenizedBlockDataset
from .modeling import count_parameters
from .sft_data import (
    CHAT_TEMPLATE,
    IGNORE_INDEX,
    LocalJsonlExamples,
    SFTTokenDataset,
    PackedSFTTokenDataset,
    collate_sft_batch,
)
from .train import (
    DistributedContext,
    autocast_context,
    cleanup_distributed,
    distributed_barrier,
    get_dtype,
    initialize_distributed,
    make_optimizer,
    make_scheduler,
    estimate_mfu,
    move_batch,
    prune_checkpoints,
    update_checkpoint_pointer,
    unwrap_model,
    wrap_distributed_model,
)


@dataclass
class SFTDatasetConfig:
    name: str | None = "HuggingFaceH4/ultrachat_200k"
    config_name: str | None = None
    split: str = "train_sft"
    eval_split: str | None = "test_sft"
    streaming: bool = True
    local_jsonl_path: str | None = None
    eval_local_jsonl_path: str | None = None
    messages_column: str = "messages"
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    prompt_column: str = "prompt"
    response_column: str = "response"
    max_examples: int | None = 50000
    eval_max_examples: int | None = 1024
    max_chars: int | None = 12000
    shuffle_buffer: int = 10000
    train_on_prompt: bool = False
    packing: bool = False
    system_prompt: str | None = "You are a helpful, concise assistant."


@dataclass
class SFTTrainerConfig:
    micro_batch_size: int = 8
    eval_micro_batch_size: int | None = None
    gradient_accumulation_steps: int = 8
    max_steps: int = 1200
    warmup_steps: int = 100
    learning_rate: float = 2e-5
    min_lr_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    lr_decay_starting_step: int | None = None
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    compile: bool = False
    liger_kernel: bool = False
    gradient_checkpointing: bool = True
    ddp_bucket_cap_mb: float = 100.0
    ddp_static_graph: bool = False
    ddp_gradient_compression: str = "none"
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 32
    save_interval: int = 200
    save_final: bool = True
    keep_last_checkpoints: int = 2
    num_workers: int = 0
    mfu_peak_tflops: float | None = None


@dataclass
class SFTReplayConfig:
    tokenized_path: str | None = None
    split: str = "train"
    ratio: float = 0.0
    seed: int = 20260822


@dataclass
class SFTConfig:
    run_name: str = "l20-edu-135m-sft"
    base_model: str = "AliceYin/l20-edu-135m"
    output_dir: str = "runs/l20-edu-135m-sft"
    seed: int = 1337
    block_size: int = 2048
    attn_implementation: str | None = "sdpa"
    dataset: SFTDatasetConfig = field(default_factory=SFTDatasetConfig)
    replay: SFTReplayConfig = field(default_factory=SFTReplayConfig)
    trainer: SFTTrainerConfig = field(default_factory=SFTTrainerConfig)

    @property
    def sequences_per_step(self) -> int:
        return self.trainer.micro_batch_size * self.trainer.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune an L20 base LM.")
    parser.add_argument("config", type=str, help="Path to an SFT YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="SFT checkpoint directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--liger-kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def apply_cli_overrides(config: SFTConfig, args: argparse.Namespace) -> SFTConfig:
    for argument, attribute in (("run_name", "run_name"), ("output_dir", "output_dir")):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config, attribute, value)
    for argument in (
        "learning_rate",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "warmup_steps",
        "eval_interval",
        "save_interval",
        "compile",
        "liger_kernel",
        "save_final",
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config.trainer, argument, value)
    return config


def load_sft_config(path: str | Path) -> SFTConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = SFTDatasetConfig(**_clean_nulls(raw.get("dataset", {})))
    replay = SFTReplayConfig(**_clean_nulls(raw.get("replay", {})))
    trainer = SFTTrainerConfig(**_clean_nulls(raw.get("trainer", {})))
    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"dataset", "replay", "trainer"} and value is not None
    }
    return SFTConfig(**top_level, dataset=dataset, replay=replay, trainer=trainer)


def save_sft_config(config: SFTConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sft_source(
    config: SFTDatasetConfig,
    *,
    split: str | None = None,
    distributed: DistributedContext | None = None,
) -> Any:
    distributed = distributed or DistributedContext(0, 0, 1, torch.device("cpu"))
    if split is not None and config.eval_local_jsonl_path and split == config.eval_split:
        return LocalJsonlExamples(
            config.eval_local_jsonl_path,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
    if config.local_jsonl_path:
        return LocalJsonlExamples(
            config.local_jsonl_path,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
    if not config.name:
        raise ValueError("SFT dataset requires either local_jsonl_path or name")

    kwargs: dict[str, Any] = {
        "path": config.name,
        "split": split or config.split,
        "streaming": config.streaming,
    }
    if config.config_name:
        kwargs["name"] = config.config_name
    dataset = load_dataset(**kwargs)
    if distributed.enabled:
        dataset = dataset.shard(
            num_shards=distributed.world_size,
            index=distributed.rank,
        )
    if config.streaming and config.shuffle_buffer > 0 and split is None:
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer, seed=0)
    return dataset


def build_sft_loader(
    config: SFTConfig,
    tokenizer: Any,
    *,
    split: str | None = None,
    max_examples: int | None = None,
    distributed: DistributedContext | None = None,
) -> DataLoader:
    distributed = distributed or DistributedContext(0, 0, 1, torch.device("cpu"))
    source = create_sft_source(
        config.dataset,
        split=split,
        distributed=distributed,
    )
    dataset = SFTTokenDataset(
        source,
        tokenizer,
        block_size=config.block_size,
        max_examples=config.dataset.max_examples if max_examples is None else max_examples,
        max_chars=config.dataset.max_chars,
        train_on_prompt=config.dataset.train_on_prompt,
        messages_column=config.dataset.messages_column,
        instruction_column=config.dataset.instruction_column,
        input_column=config.dataset.input_column,
        output_column=config.dataset.output_column,
        prompt_column=config.dataset.prompt_column,
        response_column=config.dataset.response_column,
        system_prompt=config.dataset.system_prompt,
    )
    if config.dataset.packing and split is None:
        dataset = PackedSFTTokenDataset(dataset)
    batch_size = config.trainer.micro_batch_size
    if split is not None and config.trainer.eval_micro_batch_size is not None:
        batch_size = config.trainer.eval_micro_batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_sft_batch,
        num_workers=config.trainer.num_workers,
        pin_memory=distributed.device.type == "cuda",
    )


def collate_replay_batch(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([row["input_ids"] for row in rows], dim=0)
    labels = torch.stack([row["labels"] for row in rows], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def build_replay_loader(
    config: SFTConfig,
    *,
    distributed: DistributedContext,
) -> DataLoader | None:
    ratio = config.replay.ratio
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"replay.ratio must be in [0, 1), got {ratio}")
    if ratio == 0.0:
        return None
    if not config.replay.tokenized_path:
        raise ValueError("replay.tokenized_path is required when replay.ratio > 0")
    dataset = TokenizedBlockDataset(
        config.replay.tokenized_path,
        split=config.replay.split,
        block_size=config.block_size,
        seed=config.replay.seed,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.trainer.micro_batch_size,
        collate_fn=collate_replay_batch,
        num_workers=config.trainer.num_workers,
        pin_memory=distributed.device.type == "cuda",
    )


def is_replay_step(step: int, ratio: float) -> bool:
    if step <= 0 or ratio <= 0.0:
        return False
    epsilon = 1e-12
    return math.floor(step * ratio + epsilon) > math.floor(
        (step - 1) * ratio + epsilon
    )


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: SFTConfig,
    step: int,
) -> Path:
    checkpoint_dir = Path(config.output_dir) / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_model = unwrap_model(model)
    training_use_cache = checkpoint_model.config.use_cache
    checkpoint_model.config.use_cache = True
    try:
        checkpoint_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    finally:
        checkpoint_model.config.use_cache = training_use_cache
    tokenizer.save_pretrained(checkpoint_dir)
    save_sft_config(config, checkpoint_dir / "sft_config.yaml")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    config: SFTConfig,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    distributed = distributed or DistributedContext(0, 0, 1, device)
    eval_split = config.dataset.eval_split
    if not eval_split and not config.dataset.local_jsonl_path and not config.dataset.eval_local_jsonl_path:
        return {"loss": float("nan"), "supervised_tokens": 0.0}

    model.eval()
    loader = build_sft_loader(
        config,
        tokenizer,
        split=eval_split,
        max_examples=config.dataset.eval_max_examples,
        distributed=distributed,
    )
    iterator = iter(loader)
    weighted_loss = torch.zeros((), dtype=torch.float64, device=device)
    supervised_tokens = torch.zeros((), dtype=torch.float64, device=device)
    for _ in range(config.trainer.eval_batches):
        try:
            batch = move_batch(next(iterator), device)
        except StopIteration:
            break
        batch_supervised_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
        with autocast_context(device, dtype):
            loss = model(**batch).loss
        weighted_loss += loss.detach().double() * batch_supervised_tokens
        supervised_tokens += batch_supervised_tokens
    stats = torch.stack((weighted_loss, supervised_tokens))
    if distributed.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    token_count = float(stats[1].item())
    return {
        "loss": float(stats[0].item() / token_count) if token_count else float("nan"),
        "supervised_tokens": token_count,
    }


def load_tokenizer(model_source: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def maybe_apply_liger(config: SFTConfig) -> bool:
    if not config.trainer.liger_kernel:
        return False
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama()
    return True


def estimate_sft_flops_per_token(model: torch.nn.Module, block_size: int, parameter_count: int) -> int:
    config = unwrap_model(model).config
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if hidden_size <= 0 or num_layers <= 0 or num_heads <= 0:
        return int(6 * parameter_count)
    head_dim = hidden_size // max(1, num_heads)
    attention_flops = 12 * num_layers * num_heads * head_dim * block_size
    return int(6 * parameter_count + attention_flops)


def run_training(args: argparse.Namespace, distributed: DistributedContext) -> None:
    config = apply_cli_overrides(load_sft_config(args.config), args)
    set_seed(config.seed + distributed.rank)

    output_dir = Path(config.output_dir)
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_sft_config(config, output_dir / "sft_config.yaml")
    distributed_barrier(distributed)

    device = distributed.device
    dtype = get_dtype(config.trainer.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    model_source = args.resume or config.base_model
    liger_applied = maybe_apply_liger(config)
    tokenizer = load_tokenizer(model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        attn_implementation=config.attn_implementation,
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = (
        [int(tokenizer.eos_token_id)]
        if tokenizer.eos_token_id is not None
        else []
    )
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in stop_ids:
        stop_ids.append(im_end_id)
    model.generation_config.eos_token_id = stop_ids
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if config.trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False
    model.to(device)

    if config.trainer.compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)
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

    loader = build_sft_loader(config, tokenizer, distributed=distributed)
    iterator = iter(loader)
    replay_loader = build_replay_loader(config, distributed=distributed)
    replay_iterator = iter(replay_loader) if replay_loader is not None else None
    model.train()
    parameter_count = count_parameters(unwrap_model(model))
    flops_per_token = estimate_sft_flops_per_token(model, config.block_size, parameter_count)
    peak_memory_gb = None
    peak_tflops = config.trainer.mfu_peak_tflops
    if peak_tflops is not None:
        peak_tflops *= distributed.world_size

    if distributed.is_main:
        print(
            json.dumps(
                {
                    "event": "start",
                    "run_name": config.run_name,
                    "base_model": config.base_model,
                    "device": str(device),
                    "distributed_backend": distributed.backend,
                    "world_size": distributed.world_size,
                    "dtype": config.trainer.dtype,
                    "block_size": config.block_size,
                    "attn_implementation": config.attn_implementation,
                    "sequences_per_step_per_rank": config.sequences_per_step,
                    "sequences_per_step": (
                        config.sequences_per_step * distributed.world_size
                    ),
                    "tokens_per_step": (
                        config.block_size
                        * config.sequences_per_step
                        * distributed.world_size
                    ),
                    "train_on_prompt": config.dataset.train_on_prompt,
                    "packing": config.dataset.packing,
                    "replay_ratio": config.replay.ratio,
                    "replay_tokenized_path": config.replay.tokenized_path,
                    "start_step": start_step,
                    "liger_kernel": liger_applied,
                    "parameters": parameter_count,
                    "flops_per_token_estimate": flops_per_token,
                    "mfu_peak_tflops_total": peak_tflops,
                    "ddp_bucket_cap_mb": config.trainer.ddp_bucket_cap_mb,
                    "ddp_static_graph": config.trainer.ddp_static_graph,
                    "ddp_gradient_compression": (
                        config.trainer.ddp_gradient_compression
                    ),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    last_log = time.time()
    last_log_step = start_step
    supervised_tokens_since_log = torch.zeros((), dtype=torch.float64, device=device)
    tokens_since_log = torch.zeros((), dtype=torch.float64, device=device)
    loss_since_log = torch.zeros((), dtype=torch.float64, device=device)
    sft_loss_since_log = torch.zeros((), dtype=torch.float64, device=device)
    replay_loss_since_log = torch.zeros((), dtype=torch.float64, device=device)
    sft_steps_since_log = 0
    replay_steps_since_log = 0
    last_checkpoint_dir: Path | None = None
    for step in range(start_step + 1, config.trainer.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        use_replay = replay_iterator is not None and is_replay_step(
            step, config.replay.ratio
        )
        if use_replay:
            replay_steps_since_log += 1
        for micro_step in range(config.trainer.gradient_accumulation_steps):
            try:
                batch = next(replay_iterator if use_replay else iterator)
            except StopIteration:
                if use_replay:
                    assert replay_loader is not None
                    replay_iterator = iter(replay_loader)
                    batch = next(replay_iterator)
                else:
                    iterator = iter(loader)
                    batch = next(iterator)
            batch = move_batch(batch, device)
            supervised_tokens_since_log += (batch["labels"] != IGNORE_INDEX).sum()
            tokens_since_log += batch["attention_mask"].sum()
            should_sync = micro_step == config.trainer.gradient_accumulation_steps - 1
            sync_context = (
                nullcontext()
                if should_sync or not distributed.enabled
                else model.no_sync()  # type: ignore[union-attr]
            )
            with sync_context:
                with autocast_context(device, dtype):
                    loss = model(**batch).loss / config.trainer.gradient_accumulation_steps
                total_loss += (
                    loss.detach().float()
                    * config.trainer.gradient_accumulation_steps
                )
                loss.backward()

        if config.trainer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.trainer.grad_clip,
                error_if_nonfinite=True,
            )
        optimizer.step()
        scheduler.step()
        loss_since_log += total_loss
        if use_replay:
            replay_loss_since_log += total_loss
        else:
            sft_loss_since_log += total_loss
            sft_steps_since_log += 1

        if step % config.trainer.log_interval == 0 or step == 1:
            now = time.time()
            elapsed = max(now - last_log, 1e-9)
            steps_since_log = max(1, step - last_log_step)
            last_log = now
            last_log_step = step
            stats = torch.stack(
                (
                    loss_since_log,
                    tokens_since_log,
                    supervised_tokens_since_log,
                    sft_loss_since_log,
                    replay_loss_since_log,
                )
            )
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
            if distributed.enabled:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            global_elapsed = max(float(elapsed_tensor.item()), 1e-9)
            tokens_window = float(stats[1].item())
            supervised_tokens_window = float(stats[2].item())
            tokens_per_sec = tokens_window / global_elapsed
            supervised_tokens_per_sec = supervised_tokens_window / global_elapsed
            mfu = estimate_mfu(
                tokens_per_sec=tokens_per_sec,
                flops_per_token=flops_per_token,
                peak_tflops=peak_tflops,
            )
            if device.type == "cuda":
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            payload: dict[str, Any] = {
                "event": "train",
                "step": step,
                "loss": (
                    float(stats[0].item())
                    / distributed.world_size
                    / config.trainer.gradient_accumulation_steps
                    / steps_since_log
                ),
                "lr": scheduler.get_last_lr()[0],
                "sequences": (
                    step * config.sequences_per_step * distributed.world_size
                ),
                "tokens_window": tokens_window,
                "supervised_tokens_window": supervised_tokens_window,
                "tokens_per_sec_window": tokens_per_sec,
                "supervised_tokens_per_sec_window": supervised_tokens_per_sec,
                "steps_per_sec_window": steps_since_log / global_elapsed,
                "replay_steps_window": replay_steps_since_log,
            }
            if sft_steps_since_log > 0:
                payload["sft_loss_window"] = (
                    float(stats[3].item())
                    / distributed.world_size
                    / config.trainer.gradient_accumulation_steps
                    / sft_steps_since_log
                )
            if replay_steps_since_log > 0:
                payload["replay_loss_window"] = (
                    float(stats[4].item())
                    / distributed.world_size
                    / config.trainer.gradient_accumulation_steps
                    / replay_steps_since_log
                )
            if mfu is not None:
                payload["mfu"] = mfu
                payload["mfu_pct"] = 100.0 * mfu
            if peak_memory_gb is not None:
                payload["peak_memory_gb"] = peak_memory_gb
            if distributed.is_main:
                print(json.dumps(payload, ensure_ascii=True), flush=True)
            supervised_tokens_since_log.zero_()
            tokens_since_log.zero_()
            loss_since_log.zero_()
            sft_loss_since_log.zero_()
            replay_loss_since_log.zero_()
            sft_steps_since_log = 0
            replay_steps_since_log = 0

        if config.trainer.eval_interval > 0 and step % config.trainer.eval_interval == 0:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            metrics = evaluate(
                model,
                config,
                tokenizer,
                device,
                dtype,
                distributed=distributed,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            eval_loss = metrics["loss"]
            if distributed.is_main:
                print(
                    json.dumps(
                        {
                            "event": "eval",
                            "step": step,
                            "loss": eval_loss,
                            "perplexity": (
                                float(np.exp(eval_loss))
                                if eval_loss < 20
                                else float("inf")
                            ),
                            "supervised_tokens": metrics["supervised_tokens"],
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

        if config.trainer.save_interval > 0 and step % config.trainer.save_interval == 0:
            if distributed.is_main:
                last_checkpoint_dir = save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    config,
                    step,
                )
                update_checkpoint_pointer(config.output_dir, last_checkpoint_dir)
                prune_checkpoints(config.output_dir, config.trainer.keep_last_checkpoints)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "path": str(last_checkpoint_dir),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
            distributed_barrier(distributed)

    checkpoint_dir = None
    if distributed.is_main:
        if config.trainer.save_final:
            checkpoint_dir = last_checkpoint_dir
            if checkpoint_dir is None or checkpoint_dir.name != f"step-{config.trainer.max_steps:06d}":
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
            json.dumps(
                {
                    "event": "done",
                    "checkpoint": (
                        str(checkpoint_dir) if checkpoint_dir is not None else None
                    ),
                },
                ensure_ascii=True,
            ),
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
