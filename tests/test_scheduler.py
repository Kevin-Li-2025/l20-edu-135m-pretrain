from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from l20_pretrain.config import PretrainConfig, TrainerConfig
from l20_pretrain.train import make_scheduler, override_training_steps


def make_optimizer() -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1.0)


def test_wsd_schedule_warms_stays_constant_and_decays() -> None:
    config = PretrainConfig(
        trainer=TrainerConfig(
            max_steps=100,
            warmup_steps=10,
            min_lr_ratio=0.0,
            lr_scheduler_type="wsd",
            lr_decay_starting_step=80,
        )
    )
    scheduler = make_scheduler(make_optimizer(), config)
    schedule = scheduler.lr_lambdas[0]

    assert schedule(0) == pytest.approx(0.1)
    assert schedule(9) == pytest.approx(1.0)
    assert schedule(79) == pytest.approx(1.0)
    assert schedule(90) == pytest.approx(0.5)
    assert schedule(100) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "trainer",
    [
        TrainerConfig(lr_scheduler_type="unknown"),
        TrainerConfig(lr_scheduler_type="wsd", lr_decay_starting_step=None),
        TrainerConfig(
            max_steps=100,
            warmup_steps=10,
            lr_scheduler_type="wsd",
            lr_decay_starting_step=5,
        ),
    ],
)
def test_invalid_scheduler_config_is_rejected(trainer: TrainerConfig) -> None:
    config = replace(PretrainConfig(), trainer=trainer)
    with pytest.raises(ValueError):
        make_scheduler(make_optimizer(), config)


def test_short_preflight_scales_wsd_boundaries() -> None:
    config = PretrainConfig(
        trainer=TrainerConfig(
            max_steps=10_000,
            warmup_steps=200,
            lr_scheduler_type="wsd",
            lr_decay_starting_step=8_000,
        )
    )

    short = override_training_steps(config, 3)

    assert short.trainer.max_steps == 3
    assert short.trainer.warmup_steps == 1
    assert short.trainer.lr_decay_starting_step == 2
    make_scheduler(make_optimizer(), short)
