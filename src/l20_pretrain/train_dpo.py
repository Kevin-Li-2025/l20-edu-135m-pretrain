from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import _clean_nulls


@dataclass
class DPODataConfig:
    train_jsonl_path: str = "data/posttrain/ultrafeedback_clean_v1/train.jsonl"
    eval_jsonl_path: str = "data/posttrain/ultrafeedback_clean_v1/eval.jsonl"
    max_train_examples: int | None = None
    max_eval_examples: int | None = 2000
    seed: int = 4242
    dataset_num_proc: int = 8


@dataclass
class DPOTrainerSettings:
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    beta: float = 0.5
    loss_type: str = "sigmoid"
    max_length: int = 1024
    truncation_mode: str = "keep_start"
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    precompute_ref_log_probs: bool = False
    use_liger_kernel: bool = False
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    save_final: bool = True
    dataloader_num_workers: int = 0
    ddp_bucket_cap_mb: int = 100
    entropy_chunk_tokens: int = 128


@dataclass
class DPOPipelineConfig:
    run_name: str = "l20-dpo"
    model_name_or_path: str = "AliceYin/l20-edu-135m"
    output_dir: str = "runs/l20-dpo"
    seed: int = 42
    attn_implementation: str = "sdpa"
    data: DPODataConfig = field(default_factory=DPODataConfig)
    trainer: DPOTrainerSettings = field(default_factory=DPOTrainerSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dpo_config(path: str | Path) -> DPOPipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    data = DPODataConfig(**_clean_nulls(raw.get("data", {})))
    trainer = DPOTrainerSettings(**_clean_nulls(raw.get("trainer", {})))
    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"data", "trainer"} and value is not None
    }
    return DPOPipelineConfig(**top_level, data=data, trainer=trainer)


def apply_cli_overrides(
    config: DPOPipelineConfig, args: argparse.Namespace
) -> DPOPipelineConfig:
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = args.output_dir
    trainer_overrides = {
        "max_steps": "max_steps",
        "train_batch_size": "per_device_train_batch_size",
        "eval_batch_size": "per_device_eval_batch_size",
        "eval_steps": "eval_steps",
        "save_steps": "save_steps",
        "precompute_ref_log_probs": "precompute_ref_log_probs",
        "save_final": "save_final",
    }
    for argument, attribute in trainer_overrides.items():
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config.trainer, attribute, value)
    data_overrides = {
        "max_train_examples": "max_train_examples",
        "max_eval_examples": "max_eval_examples",
    }
    for argument, attribute in data_overrides.items():
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config.data, attribute, value)
    return config


def save_dpo_config(config: DPOPipelineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def load_json_dataset(path: str, max_examples: int | None, seed: int) -> Dataset:
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.shuffle(seed=seed)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


@torch.no_grad()
def chunked_entropy_from_logits(
    logits: torch.Tensor, *, chunk_tokens: int = 128
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    vocab_size = logits.shape[-1]
    prefix_shape = logits.shape[:-1]
    entropies: list[torch.Tensor] = []
    leading_shape = logits.shape[:-2]
    indices = product(*(range(size) for size in leading_shape)) if leading_shape else [()]
    for index in indices:
        row = logits[index] if index else logits
        for start in range(0, row.shape[0], chunk_tokens):
            chunk = row[start : start + chunk_tokens].float()
            log_normalizer = torch.logsumexp(chunk, dim=-1)
            expected_logit = (torch.softmax(chunk, dim=-1) * chunk).sum(dim=-1)
            entropies.append(log_normalizer - expected_logit)
    return torch.cat(entropies).reshape(prefix_shape)


def run(config: DPOPipelineConfig) -> None:
    from trl import DPOConfig, DPOTrainer
    import trl.trainer.dpo_trainer as trl_dpo_trainer

    entropy_chunk_tokens = config.trainer.entropy_chunk_tokens

    def memory_efficient_entropy(logits: torch.Tensor) -> torch.Tensor:
        return chunked_entropy_from_logits(
            logits,
            chunk_tokens=entropy_chunk_tokens,
        )

    trl_dpo_trainer.entropy_from_logits = memory_efficient_entropy

    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dpo_config(config, output_dir / "dpo_config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.trainer.bf16 else torch.float32
    model_kwargs = {
        "dtype": dtype,
        "attn_implementation": config.attn_implementation,
    }
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = not config.trainer.gradient_checkpointing
    ref_model.config.use_cache = True

    train_dataset = load_json_dataset(
        config.data.train_jsonl_path,
        config.data.max_train_examples,
        config.data.seed,
    )
    eval_dataset = load_json_dataset(
        config.data.eval_jsonl_path,
        config.data.max_eval_examples,
        config.data.seed + 1,
    )

    args = DPOConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        seed=config.seed,
        data_seed=config.data.seed,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        max_steps=config.trainer.max_steps,
        num_train_epochs=config.trainer.num_train_epochs,
        learning_rate=config.trainer.learning_rate,
        warmup_ratio=config.trainer.warmup_ratio,
        lr_scheduler_type="cosine",
        beta=config.trainer.beta,
        loss_type=[config.trainer.loss_type],
        max_length=config.trainer.max_length,
        truncation_mode=config.trainer.truncation_mode,
        bf16=config.trainer.bf16,
        bf16_full_eval=config.trainer.bf16,
        tf32=config.trainer.tf32,
        gradient_checkpointing=config.trainer.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=config.trainer.precompute_ref_log_probs,
        use_liger_kernel=config.trainer.use_liger_kernel,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=config.trainer.logging_steps,
        eval_strategy="steps" if config.trainer.eval_steps > 0 else "no",
        eval_steps=(config.trainer.eval_steps if config.trainer.eval_steps > 0 else None),
        save_strategy="steps" if config.trainer.save_steps > 0 else "no",
        save_steps=(config.trainer.save_steps if config.trainer.save_steps > 0 else 500),
        save_total_limit=config.trainer.save_total_limit,
        dataset_num_proc=config.data.dataset_num_proc,
        dataloader_num_workers=config.trainer.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_bucket_cap_mb=config.trainer.ddp_bucket_cap_mb,
        ddp_find_unused_parameters=False,
        pad_to_multiple_of=8,
        optim="adamw_torch_fused",
        report_to=[],
        disable_tqdm=True,
        include_num_input_tokens_seen=True,
        include_tokens_per_second=True,
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "start",
                    "run_name": config.run_name,
                    "model": config.model_name_or_path,
                    "train_examples": len(train_dataset),
                    "eval_examples": len(eval_dataset),
                    "beta": config.trainer.beta,
                    "learning_rate": config.trainer.learning_rate,
                    "max_length": config.trainer.max_length,
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    final_dir = output_dir / "final"
    if config.trainer.save_final:
        trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        if config.trainer.save_final:
            tokenizer.save_pretrained(final_dir)
        print(
            json.dumps(
                {
                    "event": "done",
                    "checkpoint": str(final_dir) if config.trainer.save_final else None,
                    "metrics": result.metrics,
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-parameter DPO post-training.")
    parser.add_argument("config", help="Path to DPO YAML config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument(
        "--precompute-ref-log-probs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    run(apply_cli_overrides(load_dpo_config(args.config), args))


if __name__ == "__main__":
    main()
