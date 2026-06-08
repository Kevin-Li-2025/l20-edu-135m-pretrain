from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from .env import set_default_hf_home

set_default_hf_home()

from datasets import load_dataset
from transformers import AutoTokenizer

from .data import tokenize_without_specials
from .prepare_shards import passes_dataset_score, write_tokens
from .quality import code_quality_filter, normalize_code_text, normalize_text, quality_filter, stable_hash


@dataclass
class SourceSpec:
    name: str
    kind: str
    dataset: str
    config_name: str | None = None
    split: str = "train"
    text_column: str = "text"
    family: str = "text"
    weight: float = 1.0
    min_chars: int = 500
    max_chars: int = 45_000
    min_score: float | None = None
    min_int_score: int | None = None
    require_license_type: str | None = None
    include_metadata_header: bool = False
    download_workers: int = 8
    download_buffer_size: int = 64
    quality: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(**{key: value for key, value in raw.items() if value is not None})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mixed math/code/synthetic-textbook pretraining shards."
    )
    parser.add_argument("--recipe", required=True, help="YAML recipe with source weights and filters.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    return parser.parse_args()


def load_recipe(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle) or {}
    if not recipe.get("sources"):
        raise ValueError("Mixture recipe must define at least one source")
    return recipe


def create_s3_client() -> Any:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "Stack-Edu sources require boto3. Install project dependencies or run `pip install boto3`."
        ) from exc

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_stack_edu_blob(s3_client: Any, blob_id: str) -> str | None:
    try:
        obj = s3_client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
    except Exception as exc:
        error_code = getattr(getattr(exc, "response", None), "get", lambda *_: {})("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404"}:
            return None
        raise

    with gzip.GzipFile(fileobj=obj["Body"]) as handle:
        return handle.read().decode("utf-8", errors="ignore")


def format_code_document(example: dict[str, Any], text: str, *, include_metadata_header: bool) -> str:
    if not include_metadata_header:
        return text
    language = str(example.get("language") or "").strip()
    path = str(example.get("path") or "").strip()
    repo_name = str(example.get("repo_name") or "").strip()
    header = "\n".join(
        line
        for line in (
            f"Repository: {repo_name}" if repo_name else "",
            f"Path: {path}" if path else "",
            f"Language: {language}" if language else "",
        )
        if line
    )
    fence = language.lower().replace("csharp", "c#").replace("cpp", "cpp")
    return f"{header}\n\n```{fence}\n{text.rstrip()}\n```" if header else text


def iter_source_examples(source: SourceSpec, *, s3_client: Any | None = None) -> Iterator[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config_name:
        kwargs["name"] = source.config_name
    dataset = load_dataset(**kwargs)

    if source.kind == "stack_edu":
        if s3_client is None:
            raise ValueError("stack_edu source requires an S3 client")
        yield from iter_stack_edu_examples(dataset, source, s3_client=s3_client)
        return

    for example in dataset:
        if not isinstance(example, dict):
            continue
        if not passes_dataset_score(
            example,
            min_score=source.min_score,
            min_int_score=source.min_int_score,
        ):
            yield {"_reject": "score_reject", "_source": source.name}
            continue
        if source.require_license_type and example.get("license_type") != source.require_license_type:
            yield {"_reject": "license_reject", "_source": source.name}
            continue

        yield example


def iter_stack_edu_examples(dataset: Any, source: SourceSpec, *, s3_client: Any) -> Iterator[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> Iterator[dict[str, Any]]:
        nonlocal buffer
        if not buffer:
            return
        batch = buffer
        buffer = []
        with ThreadPoolExecutor(max_workers=max(1, source.download_workers)) as executor:
            futures = {
                executor.submit(download_stack_edu_blob, s3_client, str(example["blob_id"])): example
                for example in batch
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    text = future.result()
                except Exception as exc:
                    yield {
                        "_reject": "download_error",
                        "_source": source.name,
                        "_error": str(exc)[:500],
                    }
                    continue
                if not text:
                    yield {"_reject": "download_failed", "_source": source.name}
                    continue
                enriched = dict(example)
                enriched[source.text_column] = format_code_document(
                    enriched,
                    text,
                    include_metadata_header=source.include_metadata_header,
                )
                yield enriched

    for example in dataset:
        if not isinstance(example, dict):
            continue
        if not passes_dataset_score(
            example,
            min_score=source.min_score,
            min_int_score=source.min_int_score,
        ):
            yield {"_reject": "score_reject", "_source": source.name}
            continue
        if source.require_license_type and example.get("license_type") != source.require_license_type:
            yield {"_reject": "license_reject", "_source": source.name}
            continue
        blob_id = example.get("blob_id")
        if not isinstance(blob_id, str) or not blob_id:
            yield {"_reject": "missing_blob_id", "_source": source.name}
            continue
        buffer.append(example)
        if len(buffer) >= source.download_buffer_size:
            yield from flush_buffer()
    yield from flush_buffer()


def get_text(example: dict[str, Any], text_column: str) -> str | None:
    value = example.get(text_column)
    return value if isinstance(value, str) else None


def passes_quality(text: str, source: SourceSpec) -> tuple[bool, str]:
    if source.family == "code":
        decision = code_quality_filter(text, min_chars=source.min_chars, **source.quality)
    else:
        decision = quality_filter(text, min_chars=source.min_chars, **source.quality)
    return decision.keep, decision.reason


def compute_train_quotas(target_tokens: int, sources: list[SourceSpec]) -> dict[str, int]:
    total_weight = sum(max(0.0, source.weight) for source in sources)
    if total_weight <= 0:
        raise ValueError("At least one source must have positive weight")

    quotas: dict[str, int] = {}
    assigned = 0
    for source in sources[:-1]:
        quota = int(target_tokens * max(0.0, source.weight) / total_weight)
        quotas[source.name] = quota
        assigned += quota
    quotas[sources[-1].name] = target_tokens - assigned
    return quotas


def main() -> None:
    args = parse_args()
    recipe = load_recipe(args.recipe)
    tokenizer_name = args.tokenizer or recipe.get("tokenizer", "AliceYin/l20-edu-135m")
    output_dir = Path(args.output_dir or recipe["output_dir"])
    target_tokens = int(args.target_tokens or recipe.get("target_tokens", 300_000_000))
    val_tokens_target = int(args.val_tokens or recipe.get("val_tokens", 2_097_152))
    block_size = int(args.block_size or recipe.get("block_size", 8192))
    report_interval = int(args.report_interval or recipe.get("report_interval", 1000))
    sources = [SourceSpec.from_dict(raw) for raw in recipe["sources"]]
    quotas = compute_train_quotas(target_tokens, sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    metadata_path = output_dir / "metadata.json"
    seen: set[str] = set()
    counters: Counter[str] = Counter()
    source_counters: dict[str, Counter[str]] = {source.name: Counter() for source in sources}
    source_tokens: dict[str, int] = {source.name: 0 for source in sources}
    train_tokens = 0
    val_tokens = 0
    started_at = time.time()
    s3_client = create_s3_client() if any(source.kind == "stack_edu" for source in sources) else None

    with train_path.open("wb") as train_handle, val_path.open("wb") as val_handle:
        for source in sources:
            for example in iter_source_examples(source, s3_client=s3_client):
                counters["seen"] += 1
                source_counters[source.name]["seen"] += 1
                if "_reject" in example:
                    reason = str(example["_reject"])
                    counters[reason] += 1
                    source_counters[source.name][reason] += 1
                    continue

                raw_text = get_text(example, source.text_column)
                if not raw_text:
                    counters["empty"] += 1
                    source_counters[source.name]["empty"] += 1
                    continue

                if source.family == "code":
                    text = normalize_code_text(raw_text, max_chars=source.max_chars)
                else:
                    text = normalize_text(raw_text, max_chars=source.max_chars)
                keep, reason = passes_quality(text, source)
                if not keep:
                    key = f"quality_{reason}"
                    counters[key] += 1
                    source_counters[source.name][key] += 1
                    continue

                digest = stable_hash(text)
                if digest in seen:
                    counters["duplicate"] += 1
                    source_counters[source.name]["duplicate"] += 1
                    continue
                seen.add(digest)

                ids = tokenize_without_specials(tokenizer, text)
                if len(ids) < 64:
                    counters["too_few_tokens"] += 1
                    source_counters[source.name]["too_few_tokens"] += 1
                    continue
                ids.append(int(eos_token_id))

                if val_tokens < val_tokens_target and int(digest[:8], 16) % 97 == 0:
                    val_tokens += write_tokens(val_handle, ids)
                    source_counters[source.name]["val_kept"] += 1
                elif source_tokens[source.name] < quotas[source.name] and train_tokens < target_tokens:
                    written = write_tokens(train_handle, ids)
                    train_tokens += written
                    source_tokens[source.name] += written
                    source_counters[source.name]["train_kept"] += 1
                elif train_tokens >= target_tokens and val_tokens < val_tokens_target:
                    counters["train_full_val_scan"] += 1
                    source_counters[source.name]["train_full_val_scan"] += 1
                    continue
                else:
                    break

                counters["kept"] += 1
                source_counters[source.name]["kept"] += 1
                if counters["seen"] % report_interval == 0:
                    elapsed = max(time.time() - started_at, 1e-9)
                    print(
                        json.dumps(
                            {
                                "event": "prepare_mixture",
                                "source": source.name,
                                "seen_docs": counters["seen"],
                                "kept_docs": counters["kept"],
                                "train_tokens": train_tokens,
                                "val_tokens": val_tokens,
                                "source_train_tokens": source_tokens,
                                "tokens_per_sec": (train_tokens + val_tokens) / elapsed,
                                "rejects": {
                                    key: value
                                    for key, value in counters.items()
                                    if key not in {"seen", "kept"}
                                },
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
            if train_tokens >= target_tokens and val_tokens >= val_tokens_target:
                break

    metadata = {
        "dtype": "uint32",
        "tokenizer": tokenizer_name,
        "recipe": str(args.recipe),
        "name": recipe.get("name"),
        "block_size": block_size,
        "target_tokens": target_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_blocks": train_tokens // block_size,
        "val_blocks": val_tokens // block_size,
        "quotas": quotas,
        "source_tokens": source_tokens,
        "sources": [source.__dict__ for source in sources],
        "counters": dict(counters),
        "source_counters": {name: dict(counter) for name, counter in source_counters.items()},
        "elapsed_sec": time.time() - started_at,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps({"event": "done", **metadata}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
