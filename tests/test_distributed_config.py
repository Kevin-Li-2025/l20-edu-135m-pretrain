from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from l20_pretrain.config import PretrainConfig, TrainerConfig
from l20_pretrain.train import DistributedContext, wrap_distributed_model


def cpu_context() -> DistributedContext:
    return DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "trainer",
    [
        TrainerConfig(ddp_bucket_cap_mb=0),
        TrainerConfig(ddp_gradient_compression="invalid"),
    ],
)
def test_invalid_ddp_tuning_is_rejected(trainer: TrainerConfig) -> None:
    config = replace(PretrainConfig(), trainer=trainer)

    with pytest.raises(ValueError):
        wrap_distributed_model(torch.nn.Linear(2, 2), cpu_context(), config)


def test_valid_ddp_tuning_is_a_noop_without_distributed_context() -> None:
    model = torch.nn.Linear(2, 2)
    config = PretrainConfig(
        trainer=TrainerConfig(
            ddp_bucket_cap_mb=25,
            ddp_static_graph=True,
            ddp_gradient_compression="bf16",
        )
    )

    assert wrap_distributed_model(model, cpu_context(), config) is model
