from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .env import set_default_hf_home

set_default_hf_home()

from transformers import AutoTokenizer

from .data import create_source, tokenize_without_specials
from .config import DatasetConfig
from .quality import normalize_text, quality_filter, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, tokenize, and pack pretraining shards.")
    parser.add_argument("--output-dir", required=True, help="Directory containing train.bin, val.bin, metadata.json.")
    parser.add_argument("--tokenizer", default="AliceYin/l20-edu-135m")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config-name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--local-text-path", default=None)
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=40_000)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--min-int-score", type=int, default=3)
    parser.add_argument("--report-interval", type=int, default=1000)
    return parser.parse_args()


def get_text(example: Any, text_column: str) -> str | None:
    if isinstance(example, str):
        return example
    if isinstance(example, dict):
        value = example.get(text_column)
        return value if isinstance(value, str) else None
    return None


def passes_dataset_score(example: Any, *, min_score: float | None, min_int_score: int | None) -> bool:
    if not isinstance(example, dict):
        return True
    metadata = example.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if min_score is not None:
        score = example.get("score")
        if score is None:
            score = example.get("edu_score")
        if score is None:
            score = metadata.get("score")
        if score is None:
            score = metadata.get("edu_score")
        if score is not None and float(score) < min_score:
            return False
    if min_int_score is not None:
        int_score = example.get("int_score")
        if int_score is None:
            int_score = example.get("edu_int_score")
        if int_score is None:
            int_score = metadata.get("int_score")
        if int_score is None:
            int_score = metadata.get("edu_int_score")
        if int_score is not None and int(int_score) < min_int_score:
            return False
    return True


def write_tokens(handle: Any, ids: list[int]) -> int:
    if not ids:
        return 0
    array = np.asarray(ids, dtype=np.uint32)
    array.tofile(handle)
    return int(array.size)


def iter_examples(args: argparse.Namespace) -> Iterable[Any]:
    config = DatasetConfig(
        name=args.dataset,
        config_name=args.config_name,
        split=args.split,
        streaming=True,
        text_column=args.text_column,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_score=args.min_score,
        min_int_score=args.min_int_score,
        append_eos=True,
        shuffle_buffer=0,
        local_text_path=args.local_text_path,
    )
    return create_source(config)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    metadata_path = output_dir / "metadata.json"
    seen: set[str] = set()
    counters: Counter[str] = Counter()
    train_tokens = 0
    val_tokens = 0
    started_at = time.time()

    with train_path.open("wb") as train_handle, val_path.open("wb") as val_handle:
        for example in iter_examples(args):
            counters["seen"] += 1
            if not passes_dataset_score(
                example,
                min_score=args.min_score,
                min_int_score=args.min_int_score,
            ):
                counters["score_reject"] += 1
                continue

            raw_text = get_text(example, args.text_column)
            if not raw_text:
                counters["empty"] += 1
                continue
            text = normalize_text(raw_text, max_chars=args.max_chars)
            decision = quality_filter(text, min_chars=args.min_chars)
            if not decision.keep:
                counters[f"quality_{decision.reason}"] += 1
                continue

            digest = stable_hash(text)
            if digest in seen:
                counters["duplicate"] += 1
                continue
            seen.add(digest)

            ids = tokenize_without_specials(tokenizer, text)
            if len(ids) < 64:
                counters["too_few_tokens"] += 1
                continue
            ids.append(int(eos_token_id))

            if val_tokens < args.val_tokens and int(digest[:8], 16) % 97 == 0:
                val_tokens += write_tokens(val_handle, ids)
            else:
                train_tokens += write_tokens(train_handle, ids)
            counters["kept"] += 1

            total_tokens = train_tokens + val_tokens
            if counters["seen"] % args.report_interval == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                print(
                    json.dumps(
                        {
                            "event": "prepare",
                            "seen_docs": counters["seen"],
                            "kept_docs": counters["kept"],
                            "train_tokens": train_tokens,
                            "val_tokens": val_tokens,
                            "tokens_per_sec": total_tokens / elapsed,
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
            if train_tokens >= args.target_tokens and val_tokens >= args.val_tokens:
                break

    metadata = {
        "dtype": "uint32",
        "tokenizer": args.tokenizer,
        "dataset": args.dataset,
        "config_name": args.config_name,
        "split": args.split,
        "block_size": args.block_size,
        "target_tokens": args.target_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_blocks": train_tokens // args.block_size,
        "val_blocks": val_tokens // args.block_size,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_score": args.min_score,
            "min_int_score": args.min_int_score,
        },
        "counters": dict(counters),
        "elapsed_sec": time.time() - started_at,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps({"event": "done", **metadata}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
