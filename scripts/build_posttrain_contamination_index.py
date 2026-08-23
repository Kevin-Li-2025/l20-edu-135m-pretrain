#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi

from l20_pretrain.contamination import iter_strings


SAMPLE_TASK_RE = re.compile(r"^samples_(.+?)_\d{4}-\d{2}-\d{2}T")


def task_from_path(path: Path) -> str:
    match = SAMPLE_TASK_RE.match(path.name)
    if not match:
        raise ValueError(f"Could not infer task from sample filename: {path}")
    return match.group(1)


def iter_sample_documents(root: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(root.rglob("samples_*.jsonl")):
        task = task_from_path(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                document = payload.get("doc")
                text = " ".join(iter_strings(document)).strip()
                if text:
                    yield task, text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze held-out post-training prompts for contamination filtering."
    )
    parser.add_argument("--benchmark-samples-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--ifeval-dataset", default="google/IFEval")
    parser.add_argument("--gsm8k-dataset", default="openai/gsm8k")
    args = parser.parse_args()

    if not args.benchmark_samples_root.exists():
        raise FileNotFoundError(args.benchmark_samples_root)

    api = HfApi()
    ifeval_revision = api.dataset_info(args.ifeval_dataset).sha
    gsm8k_revision = api.dataset_info(args.gsm8k_dataset).sha

    records = list(iter_sample_documents(args.benchmark_samples_root))
    for row in load_dataset(
        args.ifeval_dataset,
        split="train",
        revision=ifeval_revision,
    ):
        records.append(("ifeval", str(row["prompt"]).strip()))
    for row in load_dataset(
        args.gsm8k_dataset,
        "main",
        split="test",
        revision=gsm8k_revision,
    ):
        records.append(("gsm8k", str(row["question"]).strip()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as handle:
        for benchmark, text in records:
            key = (benchmark, text)
            if key in seen:
                continue
            seen.add(key)
            counts[benchmark] += 1
            handle.write(
                json.dumps(
                    {"benchmark": benchmark, "text": text},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "records": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "sources": {
            "benchmark_samples_root": str(args.benchmark_samples_root),
            "ifeval": {"dataset": args.ifeval_dataset, "revision": ifeval_revision},
            "gsm8k": {"dataset": args.gsm8k_dataset, "revision": gsm8k_revision},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
