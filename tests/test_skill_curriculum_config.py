from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from l20_pretrain.train_sft import load_sft_config


ROOT = Path(__file__).parents[1]


def test_skill_curriculum_pilot_uses_replay_and_frozen_data_path() -> None:
    config = load_sft_config(
        ROOT / "configs" / "posttrain" / "skill_curriculum_sft_lr_pilot_v1.yaml"
    )

    assert config.dataset.local_jsonl_path.endswith("skill_curriculum_v1/final/train.jsonl")
    assert config.replay.ratio == pytest.approx(0.10)
    assert config.trainer.max_steps == 200
    assert config.trainer.micro_batch_size == 32


def test_zh_skill_suite_has_unique_nonempty_heldout_prompts() -> None:
    path = ROOT / "configs" / "posttrain" / "zh_skill_quality_prompts_v1.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    assert len(rows) == 16
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["messages"] and row["checks"] for row in rows)


def test_teacher_recipe_pins_model_and_job_revision() -> None:
    path = ROOT / "configs" / "posttrain" / "qwen3_bilingual_teacher_v1.yaml"
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert len(recipe["teacher"]["revision"]) == 40
    assert len(recipe["jobs"]["sha256"]) == 64
    assert recipe["generation"]["batch_size"] == 32
    assert recipe["generation"]["length_balanced_sharding"] is True
