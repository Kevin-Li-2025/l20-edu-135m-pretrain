from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "HuggingFaceFW/fineweb-edu"
    config_name: str | None = "sample-10BT"
    split: str = "train"
    streaming: bool = True
    text_column: str = "text"
    min_chars: int = 200
    max_chars: int | None = 50000
    min_score: float | None = None
    min_int_score: int | None = None
    append_eos: bool = True
    shuffle_buffer: int = 10000
    max_docs: int | None = None
    local_text_path: str | None = None


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


@dataclass
class TrainerConfig:
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 32
    max_steps: int = 1000
    warmup_steps: int = 100
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    compile: bool = False
    gradient_checkpointing: bool = False
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 64
    save_interval: int = 1000
    keep_last_checkpoints: int = 2
    num_workers: int = 0


@dataclass
class PretrainConfig:
    run_name: str = "l20-pretrain"
    output_dir: str = "runs/l20-pretrain"
    seed: int = 1337
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
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


def _clean_nulls(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def load_config(path: str | Path) -> PretrainConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = DatasetConfig(**_clean_nulls(raw.get("dataset", {})))
    model = ModelConfig(**_clean_nulls(raw.get("model", {})))
    trainer = TrainerConfig(**_clean_nulls(raw.get("trainer", {})))

    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"dataset", "model", "trainer"} and value is not None
    }
    return PretrainConfig(
        **top_level,
        dataset=dataset,
        model=model,
        trainer=trainer,
    )


def save_config(config: PretrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
