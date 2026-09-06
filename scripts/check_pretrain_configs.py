from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from l20_pretrain.config import load_config  # noqa: E402


def candidate_paths(root: Path) -> list[Path]:
    return sorted([*root.glob("*.yaml"), *(root / "configs").glob("*.yaml")])


def uses_pretrain_schema(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        value: Any = yaml.safe_load(handle)
    return isinstance(value, dict) and isinstance(value.get("model"), dict) and isinstance(
        value.get("trainer"), dict
    )


def validate_paths(paths: list[Path]) -> int:
    checked = 0
    for path in paths:
        if not uses_pretrain_schema(path):
            continue
        config = load_config(path)
        checked += 1
        print(
            f"{path}: valid; run={config.run_name}; "
            f"tokens_per_step={config.tokens_per_step:,}; planned_tokens={config.planned_tokens:,}"
        )
    if checked == 0:
        raise ValueError("no pretraining configuration files were found")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all L20 pretraining configuration files.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    checked = validate_paths(candidate_paths(root))
    print(f"Validated {checked} pretraining configurations.")


if __name__ == "__main__":
    main()
