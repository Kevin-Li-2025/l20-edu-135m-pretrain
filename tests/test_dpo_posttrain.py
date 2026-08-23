from __future__ import annotations

import json
from pathlib import Path

import torch

from l20_pretrain.train_dpo import (
    apply_cli_overrides,
    chunked_entropy_from_logits,
    load_dpo_config,
    load_json_dataset,
)


def test_load_dpo_config_uses_frozen_recipe() -> None:
    path = Path(__file__).parents[1] / "configs/posttrain/ultrafeedback_dpo_pilot_v1.yaml"

    config = load_dpo_config(path)

    assert config.trainer.beta == 0.5
    assert config.trainer.learning_rate == 1e-6
    assert config.trainer.max_length == 1024
    assert config.data.max_train_examples == 5000


def test_json_preference_subset_is_seeded_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "preferences.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": str(index)}],
                    "chosen": [{"role": "assistant", "content": "yes"}],
                    "rejected": [{"role": "assistant", "content": "no"}],
                }
            )
            + "\n"
            for index in range(20)
        ),
        encoding="utf-8",
    )

    first = load_json_dataset(str(path), 5, 42)
    second = load_json_dataset(str(path), 5, 42)

    assert len(first) == 5
    assert first["prompt"] == second["prompt"]


def test_dpo_cli_overrides_benchmark_runtime_only() -> None:
    path = Path(__file__).parents[1] / "configs/posttrain/ultrafeedback_dpo_pilot_v1.yaml"
    config = load_dpo_config(path)
    args = type(
        "Args",
        (),
        {
            "output_dir": "/tmp/benchmark",
            "max_steps": 10,
            "train_batch_size": 16,
            "eval_batch_size": None,
            "eval_steps": 0,
            "save_steps": 0,
            "max_train_examples": 512,
            "max_eval_examples": None,
            "precompute_ref_log_probs": False,
            "save_final": False,
        },
    )()

    config = apply_cli_overrides(config, args)

    assert config.output_dir == "/tmp/benchmark"
    assert config.trainer.per_device_train_batch_size == 16
    assert config.trainer.eval_steps == 0
    assert not config.trainer.save_final
    assert config.data.max_train_examples == 512


def test_chunked_entropy_matches_categorical_entropy_for_noncontiguous_logits() -> None:
    generator = torch.Generator().manual_seed(7)
    full = torch.randn(3, 9, 17, generator=generator)
    logits = full[:, :-1, :]

    actual = chunked_entropy_from_logits(logits, chunk_tokens=3)
    expected = torch.distributions.Categorical(logits=logits.float()).entropy()

    assert not logits.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
