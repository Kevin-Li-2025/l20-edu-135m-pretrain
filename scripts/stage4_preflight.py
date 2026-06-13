#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch


FILES = (
    "configs/mixtures/l20_stage4_hq_crossdedup.yaml",
    "configs/l20_edu_135m_stage4_hq_crossdedup_8k.yaml",
    "scripts/prepare_l20_stage4_hq_crossdedup_8k.sh",
    "scripts/eval_smollm_benchmark.sh",
    "scripts/run_stage4_full_autopilot.sh",
    "src/l20_pretrain/data_guard.py",
    "src/l20_pretrain/prepare_mixture_shards.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--min-free-gib", type=float, default=45.0)
    parser.add_argument("--out", default="runs/stage4-full-autopilot-state/preflight.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []
    hashes = {}
    for relative in FILES:
        path = root / relative
        if not path.is_file():
            failures.append(f"missing required file: {relative}")
        else:
            hashes[relative] = sha256(path)

    free_gib = shutil.disk_usage(root).free / 2**30
    if free_gib < args.min_free_gib:
        failures.append(f"only {free_gib:.1f} GiB free; require {args.min_free_gib:.1f} GiB")
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    token_available = bool(os.environ.get("HF_TOKEN")) or (
        Path.home() / ".cache/huggingface/token"
    ).is_file()
    if not token_available:
        failures.append("Hugging Face token is unavailable")

    packages = {}
    for name in ("torch", "transformers", "datasets", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package: {name}")
    eval_python = root / ".venv-eval/bin/python"
    if not eval_python.is_file():
        failures.append("missing .venv-eval/bin/python")
    else:
        try:
            packages["lm-eval"] = subprocess.check_output(
                [
                    str(eval_python),
                    "-c",
                    "import importlib.metadata as m; print(m.version('lm_eval'))",
                ],
                text=True,
            ).strip()
        except Exception:
            failures.append("lm-eval is unavailable in .venv-eval")
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        driver = None

    payload = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "free_gib": free_gib,
        "file_sha256": hashes,
        "packages": packages,
        "gpu": gpu,
        "nvidia_driver": driver,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
