import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from l20_pretrain.config import load_config
from l20_pretrain.data import TokenizedBlockDataset

ROOT = Path(__file__).parents[1]


def load_probe_verifier():
    spec = importlib.util.spec_from_file_location(
        "probe_verifier", ROOT / "scripts/verify_fineweb_recovery_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archived_gpu_probes_are_hash_verified():
    load_probe_verifier().verify(ROOT / "results/fineweb_recovery/probe_20260906")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAIL"),
        ("scope", "quality_win"),
        ("sampled_parameters_changed", "true"),
        ("tokens_per_step", 1),
        ("parameters", 1),
        ("seed", 1),
        ("validation_smoke_loss", float("nan")),
        ("validation_smoke_batches", 1024),
        ("peak_allocated_bytes", 0),
        ("steps", []),
    ],
)
def test_gpu_probe_semantic_gates_reject_corruption(field, value):
    verifier = load_probe_verifier()
    path = ROOT / "results/fineweb_recovery/probe_20260906/deep_cosine-1560876.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    with pytest.raises(ValueError):
        verifier.validate_receipt(receipt, "deep_cosine", 134515008, ROOT)


def test_recovery_matrix_preserves_training_budget_and_historical_sources():
    files = sorted((ROOT / "configs/fineweb_recovery").glob("*.yaml"))
    assert len(files) == 12
    runs = set()
    for path in files:
        config = load_config(path)
        assert config.tokens_per_step == 159744
        assert config.planned_tokens == 999997440
        assert config.trainer.micro_batch_size == 2
        assert config.trainer.gradient_accumulation_steps == 39
        assert config.trainer.eval_batches * config.trainer.micro_batch_size == 2048
        assert config.trainer.mfu_peak_tflops == 165.2
        assert config.trainer.gradient_checkpointing is False
        assert config.trainer.num_workers == 0
        assert config.seed in (20260906, 20260907, 20260908)
        assert config.output_dir.startswith("runs/fineweb-recovery-")
        runs.add(config.output_dir)
    assert len(runs) == 12
    original = load_config(ROOT / "configs/l20_135m_fineweb_1b.yaml")
    assert original.trainer.micro_batch_size == 6
    assert original.trainer.eval_batches == 256


def test_recovery_generator_changes_only_declared_controls():
    spec = importlib.util.spec_from_file_location(
        "recovery", ROOT / "scripts/prepare_fineweb_recovery.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for role, source_name in module.SOURCES:
        source = yaml.safe_load((ROOT / "configs" / source_name).read_text())
        for seed in module.SEEDS:
            path = ROOT / "configs/fineweb_recovery" / f"{role}_s{seed}.yaml"
            actual = yaml.safe_load(path.read_text())
            assert actual == module.recovery_config(source, role, seed)
            assert actual["model"] == source["model"]
            assert actual["dataset"] == source["dataset"]
            for key in (
                "max_steps",
                "warmup_steps",
                "learning_rate",
                "lr_schedule",
                "dtype",
            ):
                assert actual["trainer"][key] == source["trainer"][key]


def test_rebatching_preserves_seeded_blocks_per_optimizer_step(tmp_path):
    values = np.arange(1000 * 8, dtype=np.uint32)
    (tmp_path / "train.bin").write_bytes(values.tobytes())
    for seed in (20260906, 20260907, 20260908):
        loaders = [
            iter(
                torch.utils.data.DataLoader(
                    TokenizedBlockDataset(
                        tmp_path, split="train", block_size=8, seed=seed
                    ),
                    batch_size=batch_size,
                    num_workers=0,
                )
            )
            for batch_size in (6, 2)
        ]
        for _ in range(3):
            old = torch.cat([next(loaders[0])["input_ids"] for _ in range(13)])
            new = torch.cat([next(loaders[1])["input_ids"] for _ in range(39)])
            assert torch.equal(old, new)


def test_equal_accumulation_matches_fp64_toy_optimizer_update():
    # Algebraic regression only: not a claim of bitwise BF16 GPU equivalence.
    torch.manual_seed(17)
    model = torch.nn.Linear(5, 3).double()
    inputs = torch.randn(78, 5, dtype=torch.float64)
    targets = torch.randn(78, 3, dtype=torch.float64)
    updated = []
    for microbatch, accumulation in ((6, 13), (2, 39)):
        candidate = deepcopy(model)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, 78, microbatch):
            loss = (
                torch.nn.functional.mse_loss(
                    candidate(inputs[start : start + microbatch]),
                    targets[start : start + microbatch],
                )
                / accumulation
            )
            loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        updated.append([p.detach().clone() for p in candidate.parameters()])
    for old, new in zip(*updated):
        torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
