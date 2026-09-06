import importlib.util
from pathlib import Path

import yaml

from l20_pretrain.config import load_config

ROOT = Path(__file__).parents[1]


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
