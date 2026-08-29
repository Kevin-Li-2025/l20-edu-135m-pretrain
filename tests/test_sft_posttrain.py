from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from l20_pretrain.train import make_scheduler
from l20_pretrain.train_sft import (
    SFTConfig,
    SFTTrainerConfig,
    apply_cli_overrides,
    is_replay_step,
)
from l20_pretrain.sft_data import LocalJsonlExamples


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_posttrain_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_posttrain_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def make_optimizer() -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1.0)


def test_sft_config_is_compatible_with_shared_cosine_scheduler() -> None:
    config = SFTConfig(
        trainer=SFTTrainerConfig(
            max_steps=100,
            warmup_steps=10,
            min_lr_ratio=0.1,
            lr_scheduler_type="cosine",
        )
    )
    scheduler = make_scheduler(make_optimizer(), config)
    schedule = scheduler.lr_lambdas[0]

    assert schedule(0) == pytest.approx(0.1)
    assert schedule(9) == pytest.approx(1.0)
    assert schedule(100) == pytest.approx(0.1)


def test_sft_eval_batch_can_be_smaller_than_train_batch() -> None:
    config = SFTConfig(
        trainer=SFTTrainerConfig(
            micro_batch_size=32,
            eval_micro_batch_size=8,
        )
    )

    assert config.trainer.micro_batch_size == 32
    assert config.trainer.eval_micro_batch_size == 8


def test_cli_overrides_are_saved_into_runtime_config() -> None:
    config = SFTConfig()
    args = type(
        "Args",
        (),
        {
            "run_name": "pilot",
            "output_dir": "/tmp/pilot",
            "learning_rate": 3e-4,
            "micro_batch_size": 24,
            "gradient_accumulation_steps": 1,
            "max_steps": 30,
            "warmup_steps": 3,
            "eval_interval": 0,
            "save_interval": 0,
            "compile": False,
            "liger_kernel": True,
            "save_final": False,
        },
    )()

    overridden = apply_cli_overrides(config, args)

    assert overridden.run_name == "pilot"
    assert overridden.trainer.learning_rate == pytest.approx(3e-4)
    assert overridden.trainer.micro_batch_size == 24
    assert not overridden.trainer.save_final


def test_posttrain_normalization_requires_alternating_completed_dialogue() -> None:
    valid, reason = PREPARE.normalize_row(
        {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "source": "fixture",
        },
        max_chars=1000,
    )
    assert reason == "ok"
    assert valid is not None

    trailing_user, reason = PREPARE.normalize_row(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Again"},
            ]
        },
        max_chars=1000,
    )
    assert trailing_user is None
    assert reason == "invalid_conversation_boundary"


def test_deterministic_selection_is_order_independent() -> None:
    rows = [
        PREPARE.PreparedRow(
            digest=f"{index:064x}",
            messages=[
                {"role": "user", "content": str(index)},
                {"role": "assistant", "content": "ok"},
            ],
            source="fixture",
        )
        for index in range(20)
    ]

    forward = PREPARE.select_smallest(rows, 5, 7)
    backward = PREPARE.select_smallest(reversed(rows), 5, 7)

    assert [row.digest for row in forward] == [row.digest for row in backward]


def test_local_jsonl_is_disjoint_across_distributed_ranks(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(11)),
        encoding="utf-8",
    )

    shards = [
        {row["index"] for row in LocalJsonlExamples(path, rank=rank, world_size=3)}
        for rank in range(3)
    ]

    assert set.union(*shards) == set(range(11))
    assert all(
        shards[left].isdisjoint(shards[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_replay_schedule_has_exact_low_discrepancy_count() -> None:
    replay_steps = [step for step in range(1, 201) if is_replay_step(step, 0.1)]

    assert replay_steps == list(range(10, 201, 10))
