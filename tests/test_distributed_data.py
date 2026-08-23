from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from l20_pretrain.config import DatasetConfig, ModelConfig, PretrainConfig, TrainerConfig, load_config
from l20_pretrain.data import PackedTokenDataset, TokenizedBlockDataset
from l20_pretrain.train import DistributedContext, build_loader


class IntegerTokenizer:
    eos_token_id = 999

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        value = int(text)
        return {"input_ids": [value, value]}


def write_tokenized_fixture(root: Path, *, num_blocks: int, block_size: int) -> None:
    root.mkdir()
    tokens = np.arange(num_blocks * block_size, dtype=np.uint32)
    tokens.tofile(root / "train.bin")
    (root / "metadata.json").write_text(
        json.dumps({"dtype": "uint32", "block_size": block_size}),
        encoding="utf-8",
    )


def test_tokenized_blocks_are_disjoint_across_ranks(tmp_path: Path) -> None:
    block_size = 4
    num_blocks = 12
    world_size = 3
    root = tmp_path / "tokens"
    write_tokenized_fixture(root, num_blocks=num_blocks, block_size=block_size)

    rank_blocks: list[set[int]] = []
    for rank in range(world_size):
        dataset = TokenizedBlockDataset(
            root,
            split="train",
            block_size=block_size,
            seed=7,
            rank=rank,
            world_size=world_size,
        )
        iterator = iter(dataset)
        blocks_for_rank = {
            int(next(iterator)["input_ids"][0]) // block_size
            for _ in range(num_blocks // world_size)
        }
        rank_blocks.append(blocks_for_rank)

    assert set.union(*rank_blocks) == set(range(num_blocks))
    for left in range(world_size):
        for right in range(left + 1, world_size):
            assert rank_blocks[left].isdisjoint(rank_blocks[right])


def test_streaming_documents_are_sharded_before_tokenization() -> None:
    documents = [str(index) for index in range(8)]
    rank_values: list[set[int]] = []
    for rank in range(2):
        dataset = PackedTokenDataset(
            documents,
            IntegerTokenizer(),
            block_size=2,
            append_eos=False,
            rank=rank,
            world_size=2,
        )
        rank_values.append({int(row["input_ids"][0]) for row in dataset})

    assert rank_values == [{0, 2, 4, 6}, {1, 3, 5, 7}]


def test_tokenized_resume_offset_continues_same_shuffled_epoch(tmp_path: Path) -> None:
    block_size = 4
    root = tmp_path / "tokens"
    write_tokenized_fixture(root, num_blocks=20, block_size=block_size)
    full = TokenizedBlockDataset(
        root,
        split="train",
        block_size=block_size,
        seed=11,
    )
    full_iterator = iter(full)
    expected = [int(next(full_iterator)["input_ids"][0]) for _ in range(12)]

    resumed = TokenizedBlockDataset(
        root,
        split="train",
        block_size=block_size,
        seed=11,
        start_block_offset=7,
    )
    resumed_iterator = iter(resumed)
    actual = [int(next(resumed_iterator)["input_ids"][0]) for _ in range(5)]

    assert actual == expected[7:12]


def test_loader_applies_configured_offset_only_to_training_split(tmp_path: Path) -> None:
    root = tmp_path / "tokens"
    write_tokenized_fixture(root, num_blocks=20, block_size=4)
    config = PretrainConfig(
        dataset=DatasetConfig(
            tokenized_path=str(root),
            start_block_offset_per_rank=7,
        ),
        model=ModelConfig(block_size=4),
        trainer=TrainerConfig(micro_batch_size=1, num_workers=0),
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"))

    train_loader = build_loader(config, IntegerTokenizer(), distributed=context)
    eval_loader = build_loader(
        config,
        IntegerTokenizer(),
        distributed=context,
        split="val",
    )

    assert train_loader.dataset.start_block_offset == 7
    assert eval_loader.dataset.start_block_offset == 0


def test_a40_configs_keep_global_batch_near_one_million_tokens() -> None:
    five_gpu = load_config("configs/a40_5x_l20_edu_135m_12b.yaml")
    six_gpu = load_config("configs/a40_6x_l20_edu_135m_12b.yaml")

    assert five_gpu.tokens_per_step * 5 == 983_040
    assert six_gpu.tokens_per_step * 6 == 1_032_192
    assert five_gpu.tokens_per_step * 5 * five_gpu.trainer.max_steps >= 12_000_000_000
    assert six_gpu.tokens_per_step * 6 * six_gpu.trainer.max_steps >= 12_000_000_000


def test_2b_continuation_starts_after_pilot_data_without_changing_eval() -> None:
    continuation = load_config(
        "configs/a40_5x_smollm2_continuation_2b_1k_repair_lr3e4.yaml"
    )

    assert continuation.dataset.start_block_offset_per_rank == 510 * 4 * 48
    assert continuation.tokens_per_step * 5 == 983_040
    assert continuation.planned_tokens * 5 == 2_000_486_400
    assert continuation.trainer.warmup_steps == 102
    assert continuation.trainer.lr_decay_starting_step == 1_628
