from pathlib import Path

import pytest

from l20_pretrain.config import load_config


def test_token_budget() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.tokens_per_step == 256
    assert config.planned_tokens == 1280


def test_architecture_ablation_uses_matching_data_budget() -> None:
    deepthin = load_config("configs/l20_135m_deepthin.yaml")
    wide = load_config("configs/l20_wide_140m_baseline.yaml")

    assert deepthin.tokenizer_name == wide.tokenizer_name
    assert deepthin.tokenizer_revision == wide.tokenizer_revision
    assert deepthin.dataset.name == wide.dataset.name
    assert deepthin.dataset.config_name == wide.dataset.config_name
    assert deepthin.dataset.revision == wide.dataset.revision
    assert deepthin.dataset.min_score == wide.dataset.min_score
    assert deepthin.dataset.min_int_score == wide.dataset.min_int_score
    assert deepthin.model.block_size == wide.model.block_size

    ratio = deepthin.planned_tokens / wide.planned_tokens
    assert 0.995 <= ratio <= 1.005


def test_invalid_attention_shape_fails_before_training(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
model:
  hidden_size: 130
  num_attention_heads: 8
  num_key_value_heads: 3
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hidden_size.*num_attention_heads") as exc_info:
        load_config(config_path)

    assert "num_attention_heads must be divisible by model.num_key_value_heads" in str(exc_info.value)


def test_invalid_schedule_reports_all_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-schedule.yaml"
    config_path.write_text(
        """
trainer:
  max_steps: 10
  warmup_steps: 11
  min_lr_ratio: 1.5
  log_interval: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_config(config_path)

    message = str(exc_info.value)
    assert "warmup_steps must not exceed" in message
    assert "min_lr_ratio must be in [0, 1]" in message
    assert "log_interval must be greater than zero" in message


def test_streaming_evaluation_requires_explicit_independent_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "streaming-eval.yaml"
    config_path.write_text("trainer:\n  eval_interval: 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires an explicit eval_dataset"):
        load_config(config_path)


def test_evaluation_dataset_must_not_match_training_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "leaked-eval.yaml"
    config_path.write_text(
        """
dataset:
  name: example/corpus
  config_name: clean
  split: train
eval_dataset:
  name: example/corpus
  config_name: clean
  split: train
trainer:
  eval_interval: 10
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not resolve to the training dataset"):
        load_config(config_path)


def test_evaluation_dataset_must_be_a_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-eval.yaml"
    config_path.write_text("eval_dataset: validation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="eval_dataset must be a mapping"):
        load_config(config_path)
