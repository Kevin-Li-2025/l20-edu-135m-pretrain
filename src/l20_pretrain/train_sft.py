from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .env import set_default_hf_home

set_default_hf_home()

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import _clean_nulls
from .sft_data import IGNORE_INDEX, SFTTokenDataset, collate_sft_batch, iter_local_jsonl
from .train import (
    autocast_context,
    get_device,
    get_dtype,
    make_optimizer,
    make_scheduler,
    move_batch,
    prune_checkpoints,
    update_checkpoint_pointer,
    unwrap_model,
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
    system_prompt: str | None = "You are a helpful, concise assistant."


@dataclass
class SFTTrainerConfig:
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_steps: int = 1200
    warmup_steps: int = 100
    learning_rate: float = 2e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    compile: bool = False
    gradient_checkpointing: bool = True
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 32
    save_interval: int = 200
    keep_last_checkpoints: int = 2
    num_workers: int = 0


@dataclass
class SFTConfig:
    run_name: str = "l20-edu-135m-sft"
    base_model: str = "AliceYin/l20-edu-135m"
    output_dir: str = "runs/l20-edu-135m-sft"
    seed: int = 1337
    block_size: int = 2048
    dataset: SFTDatasetConfig = field(default_factory=SFTDatasetConfig)
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
    return parser.parse_args()


def load_sft_config(path: str | Path) -> SFTConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = SFTDatasetConfig(**_clean_nulls(raw.get("dataset", {})))
    trainer = SFTTrainerConfig(**_clean_nulls(raw.get("trainer", {})))
    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"dataset", "trainer"} and value is not None
    }
    return SFTConfig(**top_level, dataset=dataset, trainer=trainer)


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


def create_sft_source(config: SFTDatasetConfig, *, split: str | None = None) -> Any:
    if split is not None and config.eval_local_jsonl_path and split == config.eval_split:
        return iter_local_jsonl(config.eval_local_jsonl_path)
    if config.local_jsonl_path:
        return iter_local_jsonl(config.local_jsonl_path)
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
    if config.streaming and config.shuffle_buffer > 0 and split is None:
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer, seed=0)
    return dataset


def build_sft_loader(
    config: SFTConfig,
    tokenizer: Any,
    *,
    split: str | None = None,
    max_examples: int | None = None,
) -> DataLoader:
    source = create_sft_source(config.dataset, split=split)
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
    return DataLoader(
        dataset,
        batch_size=config.trainer.micro_batch_size,
        collate_fn=collate_sft_batch,
        num_workers=config.trainer.num_workers,
        pin_memory=torch.cuda.is_available(),
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
    unwrap_model(model).save_pretrained(checkpoint_dir, safe_serialization=True)
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
) -> dict[str, float]:
    eval_split = config.dataset.eval_split
    if not eval_split and not config.dataset.local_jsonl_path and not config.dataset.eval_local_jsonl_path:
        return {"loss": float("nan"), "supervised_tokens": 0.0}

    model.eval()
    loader = build_sft_loader(
        config,
        tokenizer,
        split=eval_split,
        max_examples=config.dataset.eval_max_examples,
    )
    iterator = iter(loader)
    losses: list[float] = []
    supervised_tokens = 0
    for _ in range(config.trainer.eval_batches):
        try:
            batch = move_batch(next(iterator), device)
        except StopIteration:
            break
        supervised_tokens += int((batch["labels"] != IGNORE_INDEX).sum().item())
        with autocast_context(device, dtype):
            loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "supervised_tokens": float(supervised_tokens),
    }


def load_tokenizer(model_source: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    config = load_sft_config(args.config)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sft_config(config, output_dir / "sft_config.yaml")

    device = get_device(args.device)
    dtype = get_dtype(config.trainer.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    model_source = args.resume or config.base_model
    tokenizer = load_tokenizer(model_source)
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
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

    loader = build_sft_loader(config, tokenizer)
    iterator = iter(loader)
    model.train()

    print(
        json.dumps(
            {
                "event": "start",
                "run_name": config.run_name,
                "base_model": config.base_model,
                "device": str(device),
                "dtype": config.trainer.dtype,
                "block_size": config.block_size,
                "sequences_per_step": config.sequences_per_step,
                "train_on_prompt": config.dataset.train_on_prompt,
                "start_step": start_step,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    last_log = time.time()
    last_log_step = start_step
    supervised_tokens_since_log = 0
    for step in range(start_step + 1, config.trainer.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        supervised_tokens = 0
        for _ in range(config.trainer.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            supervised_tokens += int((batch["labels"] != IGNORE_INDEX).sum().item())
            with autocast_context(device, dtype):
                loss = model(**batch).loss / config.trainer.gradient_accumulation_steps
            total_loss += float(loss.detach().cpu()) * config.trainer.gradient_accumulation_steps
            loss.backward()

        supervised_tokens_since_log += supervised_tokens
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
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "loss": total_loss / config.trainer.gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "sequences": step * config.sequences_per_step,
                        "supervised_tokens": supervised_tokens,
                        "supervised_tokens_per_sec_window": supervised_tokens_since_log / elapsed,
                        "steps_per_sec_window": steps_since_log / elapsed,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            supervised_tokens_since_log = 0

        if config.trainer.eval_interval > 0 and step % config.trainer.eval_interval == 0:
            metrics = evaluate(model, config, tokenizer, device, dtype)
            eval_loss = metrics["loss"]
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "step": step,
                        "loss": eval_loss,
                        "perplexity": float(np.exp(eval_loss)) if eval_loss < 20 else float("inf"),
                        "supervised_tokens": metrics["supervised_tokens"],
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
    print(json.dumps({"event": "done", "checkpoint": str(checkpoint_dir)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
