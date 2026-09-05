from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "HuggingFaceFW/fineweb-edu"
    config_name: str | None = "sample-10BT"
    revision: str | None = None
    split: str = "train"
    streaming: bool = True
    text_column: str = "text"
    tokenized_path: str | None = None
    min_chars: int = 200
    max_chars: int | None = 50000
    min_score: float | None = None
    min_int_score: int | None = None
    append_eos: bool = True
    shuffle_buffer: int = 10000
    max_docs: int | None = None
    local_text_path: str | None = None
    require_manifest: bool = False
    allow_repetition: bool = True


@dataclass
class ModelConfig:
    block_size: int = 2048
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    vocab_multiple: int = 64
    attn_implementation: str | None = "sdpa"
    rope_scaling: dict[str, Any] | None = None


@dataclass
class TrainerConfig:
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 32
    max_steps: int = 1000
    warmup_steps: int = 100
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    lr_schedule: str = "cosine"
    decay_fraction: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    deterministic: bool = False
    compile: bool = False
    compile_mode: str | None = None
    compile_fullgraph: bool | None = None
    liger_kernel: bool = False
    gradient_checkpointing: bool = False
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 64
    save_interval: int = 1000
    keep_last_checkpoints: int = 2
    num_workers: int = 0
    mfu_peak_tflops: float | None = None


@dataclass
class PretrainConfig:
    run_name: str = "l20-pretrain"
    output_dir: str = "runs/l20-pretrain"
    seed: int = 1337
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"
    tokenizer_revision: str | None = None
    init_model_name_or_path: str | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval_dataset: DatasetConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @property
    def tokens_per_step(self) -> int:
        return (
            self.model.block_size
            * self.trainer.micro_batch_size
            * self.trainer.gradient_accumulation_steps
        )

    @property
    def planned_tokens(self) -> int:
        return self.tokens_per_step * self.trainer.max_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validation_errors(self) -> list[str]:
        errors: list[str] = []

        def positive(name: str, value: int | float) -> None:
            if value <= 0:
                errors.append(f"{name} must be greater than zero (got {value!r})")

        def non_negative(name: str, value: int | float) -> None:
            if value < 0:
                errors.append(f"{name} must be non-negative (got {value!r})")

        if not self.run_name.strip():
            errors.append("run_name must not be empty")
        if not self.output_dir.strip():
            errors.append("output_dir must not be empty")
        if not self.tokenizer_name.strip():
            errors.append("tokenizer_name must not be empty")

        def validate_dataset(prefix: str, dataset: DatasetConfig) -> None:
            if not dataset.split.strip():
                errors.append(f"{prefix}.split must not be empty")
            if not dataset.text_column.strip():
                errors.append(f"{prefix}.text_column must not be empty")
            non_negative(f"{prefix}.min_chars", dataset.min_chars)
            non_negative(f"{prefix}.shuffle_buffer", dataset.shuffle_buffer)
            if dataset.max_chars is not None and dataset.max_chars < dataset.min_chars:
                errors.append(f"{prefix}.max_chars must be at least {prefix}.min_chars")
            if dataset.max_docs is not None:
                positive(f"{prefix}.max_docs", dataset.max_docs)
            if dataset.tokenized_path and dataset.local_text_path:
                errors.append(
                    f"{prefix}.tokenized_path and {prefix}.local_text_path are mutually exclusive"
                )
            if dataset.require_manifest and not dataset.tokenized_path:
                errors.append(f"{prefix}.require_manifest requires {prefix}.tokenized_path")
            if not dataset.allow_repetition and not dataset.tokenized_path:
                errors.append(f"{prefix}.allow_repetition=false requires {prefix}.tokenized_path")

        validate_dataset("dataset", self.dataset)
        if self.eval_dataset is not None:
            validate_dataset("eval_dataset", self.eval_dataset)

        positive("model.block_size", self.model.block_size)
        positive("model.hidden_size", self.model.hidden_size)
        positive("model.intermediate_size", self.model.intermediate_size)
        positive("model.num_hidden_layers", self.model.num_hidden_layers)
        positive("model.num_attention_heads", self.model.num_attention_heads)
        positive("model.num_key_value_heads", self.model.num_key_value_heads)
        positive("model.rope_theta", self.model.rope_theta)
        positive("model.rms_norm_eps", self.model.rms_norm_eps)
        positive("model.vocab_multiple", self.model.vocab_multiple)
        if self.model.num_attention_heads > 0 and self.model.hidden_size % self.model.num_attention_heads:
            errors.append("model.hidden_size must be divisible by model.num_attention_heads")
        if self.model.num_key_value_heads > 0 and self.model.num_attention_heads % self.model.num_key_value_heads:
            errors.append("model.num_attention_heads must be divisible by model.num_key_value_heads")
        if not 0.0 <= self.model.attention_dropout < 1.0:
            errors.append("model.attention_dropout must be in [0, 1)")

        positive("trainer.micro_batch_size", self.trainer.micro_batch_size)
        positive("trainer.gradient_accumulation_steps", self.trainer.gradient_accumulation_steps)
        positive("trainer.max_steps", self.trainer.max_steps)
        non_negative("trainer.warmup_steps", self.trainer.warmup_steps)
        positive("trainer.learning_rate", self.trainer.learning_rate)
        non_negative("trainer.weight_decay", self.trainer.weight_decay)
        positive("trainer.grad_clip", self.trainer.grad_clip)
        positive("trainer.log_interval", self.trainer.log_interval)
        non_negative("trainer.eval_interval", self.trainer.eval_interval)
        non_negative("trainer.eval_batches", self.trainer.eval_batches)
        non_negative("trainer.save_interval", self.trainer.save_interval)
        positive("trainer.keep_last_checkpoints", self.trainer.keep_last_checkpoints)
        non_negative("trainer.num_workers", self.trainer.num_workers)
        if self.trainer.warmup_steps > self.trainer.max_steps:
            errors.append("trainer.warmup_steps must not exceed trainer.max_steps")
        if not 0.0 <= self.trainer.min_lr_ratio <= 1.0:
            errors.append("trainer.min_lr_ratio must be in [0, 1]")
        if self.trainer.lr_schedule not in {"cosine", "wsd"}:
            errors.append("trainer.lr_schedule must be cosine or wsd")
        if not 0.0 < self.trainer.decay_fraction <= 1.0:
            errors.append("trainer.decay_fraction must be in (0, 1]")
        if not 0.0 < self.trainer.beta1 < 1.0:
            errors.append("trainer.beta1 must be in (0, 1)")
        if not 0.0 < self.trainer.beta2 < 1.0:
            errors.append("trainer.beta2 must be in (0, 1)")
        if self.trainer.dtype not in {"float32", "float16", "bfloat16"}:
            errors.append("trainer.dtype must be float32, float16, or bfloat16")
        if self.trainer.mfu_peak_tflops is not None:
            positive("trainer.mfu_peak_tflops", self.trainer.mfu_peak_tflops)

        if self.eval_dataset is not None and _dataset_identity(
            self.dataset
        ) == _dataset_identity(self.eval_dataset):
            errors.append("eval_dataset must not resolve to the training dataset")
        if (
            self.trainer.eval_interval > 0
            and self.eval_dataset is None
            and not self.dataset.tokenized_path
        ):
            errors.append(
                "trainer.eval_interval requires an explicit eval_dataset for streaming or text data"
            )

        return errors

    def validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            detail = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"Invalid pretraining configuration:\n{detail}")


def _clean_nulls(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _dataset_identity(config: DatasetConfig) -> tuple[str, ...]:
    if config.tokenized_path:
        return ("tokenized", str(Path(config.tokenized_path).expanduser().resolve()), config.split)
    if config.local_text_path:
        return ("local", str(Path(config.local_text_path).expanduser().resolve()))
    return (
        "hub",
        config.name,
        config.config_name or "",
        config.split,
    )


def load_config(path: str | Path) -> PretrainConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = DatasetConfig(**_clean_nulls(raw.get("dataset", {})))
    eval_dataset_raw = raw.get("eval_dataset")
    if eval_dataset_raw is not None and not isinstance(eval_dataset_raw, dict):
        raise ValueError("eval_dataset must be a mapping when provided")
    eval_dataset = (
        DatasetConfig(**_clean_nulls(eval_dataset_raw))
        if isinstance(eval_dataset_raw, dict)
        else None
    )
    model = ModelConfig(**_clean_nulls(raw.get("model", {})))
    trainer = TrainerConfig(**_clean_nulls(raw.get("trainer", {})))

    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"dataset", "eval_dataset", "model", "trainer"} and value is not None
    }
    config = PretrainConfig(
        **top_level,
        dataset=dataset,
        eval_dataset=eval_dataset,
        model=model,
        trainer=trainer,
    )
    config.validate()
    return config


def save_config(config: PretrainConfig, path: str | Path) -> None:
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate L20 pretraining YAML before allocating a training run."
    )
    parser.add_argument("configs", nargs="+", type=Path, help="Pretraining config YAML path(s).")
    args = parser.parse_args()

    for path in args.configs:
        config = load_config(path)
        print(
            f"{path}: valid; run={config.run_name}; "
            f"tokens_per_step={config.tokens_per_step:,}; planned_tokens={config.planned_tokens:,}"
        )


if __name__ == "__main__":
    main()
