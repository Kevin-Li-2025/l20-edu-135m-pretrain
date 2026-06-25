#!/usr/bin/env python3
"""Lightweight public-release hygiene checks for the repository."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


SECRET_PATTERNS = {
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
}

FORBIDDEN_PARTS = {
    ".venv",
    "venv",
    "runs",
    "checkpoints",
    "logs",
    "wandb",
    "eval_results",
    ".nvidia-user-packages",
}

FORBIDDEN_SUFFIXES = {
    ".bak",
    ".tmp",
    ".pt",
    ".pth",
    ".safetensors",
}

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "docs/README.md",
    "docs/reproducibility.md",
    "docs/project_report/TECHNICAL_REPORT.md",
    "docs/project_report/ablation_plan.json",
    "paper/l20_edu_135m_arxiv.tex",
    "paper/l20_edu_135m_arxiv.pdf",
    "results/README.md",
    "results/benchmark_comparison.csv",
    "results/stage4/final_model.json",
    "results/rlvr/gsm8k_before_summary.json",
    "scripts/check_ablation_plan.py",
    "src/l20_pretrain/rlvr_rewards.py",
]


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            files.append(path)
    return files


def is_text_candidate(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def check_required(root: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            errors.append(f"missing required file: {name}")
    return errors


def check_forbidden_files(root: Path, files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        parts = set(path.relative_to(root).parts)
        name = path.name
        if parts & FORBIDDEN_PARTS:
            errors.append(f"forbidden path committed: {rel(path, root)}")
        if any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            if not rel(path, root).startswith("paper/"):
                errors.append(f"forbidden artifact suffix: {rel(path, root)}")
        if name.startswith("tmp_") or name.startswith("tmp-"):
            errors.append(f"temporary file committed: {rel(path, root)}")
    return errors


def check_secrets(root: Path, files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {rel(path, root)}")
    return errors


def check_size(root: Path, files: list[Path], max_mb: int) -> list[str]:
    errors: list[str] = []
    max_bytes = max_mb * 1024 * 1024
    for path in files:
        if path.stat().st_size > max_bytes:
            errors.append(f"large file exceeds {max_mb} MiB: {rel(path, root)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--max-mb", type=int, default=25, help="maximum tracked file size")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = iter_files(root)
    errors: list[str] = []
    errors.extend(check_required(root))
    errors.extend(check_forbidden_files(root, files))
    errors.extend(check_secrets(root, files))
    errors.extend(check_size(root, files, args.max_mb))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"repo hygiene ok: {len(files)} files scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
