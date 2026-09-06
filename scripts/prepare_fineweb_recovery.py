"""Materialize a new, matched memory-safe matrix; never edit historical configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

SEEDS = (20260906, 20260907, 20260908)
SOURCES = (
    ("deep_cosine", "l20_135m_fineweb_1b.yaml"),
    ("deep_wsd", "l20_135m_fineweb_wsd_1b.yaml"),
    ("wide_cosine", "l20_140m_wide_fineweb_1b.yaml"),
    ("wide_wsd", "l20_140m_wide_fineweb_wsd_1b.yaml"),
)


def recovery_config(source: dict, role: str, seed: int) -> dict:
    config = copy.deepcopy(source)
    config["seed"] = seed
    config["run_name"] = f"fineweb-recovery-{role}-s{seed}"
    config["output_dir"] = f"runs/{config['run_name']}"
    config["trainer"].update(
        micro_batch_size=2,
        gradient_accumulation_steps=39,
        # All 2048 complete validation blocks, not a seed-dependent subset.
        eval_batches=1024,
        mfu_peak_tflops=165.2,
        keep_last_checkpoints=1,
        save_interval=3130,
    )
    return config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "configs/fineweb_recovery"
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for role, filename in SOURCES:
            source = yaml.safe_load((root / "configs" / filename).read_text())
            target = output / f"{role}_s{seed}.yaml"
            content = yaml.safe_dump(
                recovery_config(source, role, seed), sort_keys=False
            )
            if target.exists() and target.read_text() != content:
                raise FileExistsError(
                    f"refusing to replace a different recovery config: {target}"
                )
            target.write_text(content)
            print(target.relative_to(root))


if __name__ == "__main__":
    main()
