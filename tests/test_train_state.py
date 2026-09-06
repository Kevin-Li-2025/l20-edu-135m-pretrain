import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from l20_pretrain.config import DatasetConfig, ModelConfig, PretrainConfig, TrainerConfig
from l20_pretrain.data import TokenizedBlockDataset, collate_batch
from l20_pretrain.train import (
    capture_rng_state,
    load_or_create_model,
    make_scheduler,
    restore_rng_state,
    restore_trainer_state,
    set_seed,
)


def test_resume_preserves_checkpoint_parameter_dtype(monkeypatch) -> None:
    config = _make_config("unused")
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_from_pretrained(path: str, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "l20_pretrain.train.configure_pretrained_model_config",
        lambda _config, _path: object(),
    )
    monkeypatch.setattr(
        "l20_pretrain.train.AutoModelForCausalLM.from_pretrained",
        fake_from_pretrained,
    )

    model = load_or_create_model(config, object(), "checkpoint", torch.bfloat16)

    assert model is sentinel
    assert "dtype" not in captured
    assert "torch_dtype" not in captured


def test_rng_state_round_trip() -> None:
    set_seed(29)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.random()
    np.random.random()
    torch.rand(4)
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


class TinyDropoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs.float()).square().mean()


def _make_config(tokenized_path: str) -> PretrainConfig:
    return PretrainConfig(
        dataset=DatasetConfig(tokenized_path=tokenized_path, streaming=False),
        model=ModelConfig(block_size=4),
        trainer=TrainerConfig(
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=6,
            warmup_steps=1,
            num_workers=0,
            eval_interval=0,
        ),
    )


def _train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loader: DataLoader,
    steps: int,
) -> None:
    iterator = iter(loader)
    for _ in range(steps):
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch["input_ids"])
        loss.backward()
        optimizer.step()
        scheduler.step()


def _loader(path, *, seed: int, offset: int = 0) -> DataLoader:
    dataset = TokenizedBlockDataset(
        path,
        split="train",
        block_size=4,
        seed=seed,
        start_block_offset=offset,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_batch,
        generator=generator,
    )


def test_interrupted_resume_matches_continuous_training(tmp_path) -> None:
    values = np.arange(160, dtype=np.uint32) % 17
    (tmp_path / "train.bin").write_bytes(values.tobytes())
    config = _make_config(str(tmp_path))

    set_seed(config.seed)
    continuous_model = TinyDropoutModel()
    continuous_optimizer = torch.optim.AdamW(continuous_model.parameters(), lr=1e-3)
    continuous_scheduler = make_scheduler(continuous_optimizer, config)
    _train_steps(
        continuous_model,
        continuous_optimizer,
        continuous_scheduler,
        _loader(tmp_path, seed=config.seed),
        6,
    )

    set_seed(config.seed)
    split_model = TinyDropoutModel()
    split_optimizer = torch.optim.AdamW(split_model.parameters(), lr=1e-3)
    split_scheduler = make_scheduler(split_optimizer, config)
    _train_steps(
        split_model,
        split_optimizer,
        split_scheduler,
        _loader(tmp_path, seed=config.seed),
        3,
    )
    state_path = tmp_path / "trainer_state.pt"
    torch.save(
        {
            "step": 3,
            "optimizer": split_optimizer.state_dict(),
            "scheduler": split_scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "consumed_train_blocks": 6,
        },
        state_path,
    )

    resumed_model = TinyDropoutModel()
    resumed_model.load_state_dict(split_model.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = make_scheduler(resumed_optimizer, config)
    start_step, offset = restore_trainer_state(
        state_path,
        resumed_optimizer,
        resumed_scheduler,
        config,
    )
    _train_steps(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        _loader(tmp_path, seed=config.seed, offset=offset),
        config.trainer.max_steps - start_step,
    )

    for expected, actual in zip(continuous_model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)
