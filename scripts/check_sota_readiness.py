#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.config import load_config


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def gpu_info() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return {"available": bool(lines), "gpus": lines}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SOTA pipeline readiness.")
    parser.add_argument("--manifest", default="configs/sota_eval.yaml")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    candidate = load_config(manifest["candidate"]["config"])
    controlled = load_config(manifest["controlled_baseline"]["config"])
    tolerance = float(manifest["gate"]["token_budget_tolerance"])
    budget_ratio = candidate.planned_tokens / controlled.planned_tokens

    checks = {
        "same_tokenizer": candidate.tokenizer_name == controlled.tokenizer_name,
        "same_dataset": candidate.dataset.name == controlled.dataset.name
        and candidate.dataset.config_name == controlled.dataset.config_name,
        "same_filter": candidate.dataset.min_score == controlled.dataset.min_score
        and candidate.dataset.min_int_score == controlled.dataset.min_int_score
        and candidate.dataset.min_chars == controlled.dataset.min_chars,
        "same_context": candidate.model.block_size == controlled.model.block_size,
        "token_budget_within_tolerance": abs(budget_ratio - 1.0) <= tolerance,
        "python_deps": all(
            module_available(name)
            for name in ["torch", "transformers", "datasets", "yaml", "numpy"]
        ),
        "lm_eval_available": module_available("lm_eval"),
        "gpu": gpu_info(),
    }
    ok = all(value is True for key, value in checks.items() if key != "gpu")

    payload = {
        "status": "ready" if ok else "blocked",
        "budget_ratio": budget_ratio,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
