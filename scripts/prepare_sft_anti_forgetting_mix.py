#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterator
from urllib.parse import urlsplit

from datasets import load_dataset
import requests
from transformers import AutoTokenizer

from l20_pretrain.data_guard import CrossSourceDataGuard
from l20_pretrain.prepare_mixture_shards import download_parquet_to_local_cache


BAD_PHRASES = (
    "as an ai language model",
    "i am an ai language model",
    "i cannot browse",
    "i don't have access to",
    "i do not have access to",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a decontaminated Smol-SmolTalk SFT dataset."
    )
    parser.add_argument("--sft-dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--sft-split", default="train")
    parser.add_argument("--sft-eval-split", default="test")
    parser.add_argument("--sft-source-limit", type=int, default=500_000)
    parser.add_argument("--tokenizer", default="AliceYin/l20-edu-135m")
    parser.add_argument(
        "--target-size",
        type=int,
        default=0,
        help="Maximum accepted training rows; 0 keeps the full filtered split.",
    )
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--min-response-chars", type=int, default=20)
    parser.add_argument("--max-response-chars", type=int, default=16_000)
    parser.add_argument("--max-example-chars", type=int, default=50_000)
    parser.add_argument("--max-example-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument(
        "--contamination-path",
        default="data/benchmark_contamination/eval_5tasks.jsonl",
    )
    parser.add_argument(
        "--dedup-index",
        default="data/sft/smol_smoltalk_guard.sqlite",
    )
    parser.add_argument(
        "--parquet-cache-dir",
        default="data/hf_parquet_cache",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_messages(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = text(item.get("role")).lower()
        content = text(item.get("content"))
        if role not in {"system", "user", "assistant"} or not content:
            return None
        messages.append({"role": role, "content": content})
    return messages or None


def standardize_example(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = parse_messages(example.get("messages"))
    if not messages:
        return None
    system_positions = [index for index, message in enumerate(messages) if message["role"] == "system"]
    if system_positions and system_positions != [0]:
        return None
    conversation = messages[1:] if messages[0]["role"] == "system" else messages
    if not conversation or conversation[0]["role"] != "user":
        return None
    expected = "user"
    for message in conversation:
        if message["role"] != expected:
            return None
        expected = "assistant" if expected == "user" else "user"
    if conversation[-1]["role"] != "assistant":
        return None
    return {
        "messages": messages,
        "source": text(example.get("source")) or "unknown",
    }


def joined_content(row: dict[str, Any]) -> str:
    return "\n\n".join(
        message["content"]
        for message in row["messages"]
        if message["role"] in {"user", "assistant"}
    )


def chatml_text(row: dict[str, Any]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in row["messages"]
    )


def stable_int(value: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def repetition_penalty(response: str) -> float:
    words = re.findall(r"[a-zA-Z0-9']+", response.lower())
    if len(words) < 20:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    lines = [line.strip().lower() for line in response.splitlines() if line.strip()]
    repeated_lines = len(lines) - len(set(lines))
    return max(0.0, 0.40 - unique_ratio) * 2.0 + min(1.0, repeated_lines * 0.15)


def basic_quality_reason(
    row: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
) -> str | None:
    serialized = json.dumps(row, ensure_ascii=False)
    if len(serialized) > args.max_example_chars:
        return "too_many_chars"
    assistants = [
        message["content"] for message in row["messages"] if message["role"] == "assistant"
    ]
    if not assistants:
        return "missing_assistant"
    for response in assistants:
        if len(response) < args.min_response_chars:
            return "response_too_short"
        if len(response) > args.max_response_chars:
            return "response_too_long"
        lowered = response.lower()
        if any(phrase in lowered for phrase in BAD_PHRASES):
            return "model_disclaimer"
        if repetition_penalty(response) >= 0.8:
            return "response_repetition"
    token_count = len(
        tokenizer(
            chatml_text(row),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    if token_count > args.max_example_tokens:
        return "too_many_tokens"
    return None


def converted_parquet_urls(dataset: str, split: str) -> list[tuple[str, str]]:
    response = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset},
        timeout=(10, 60),
    )
    response.raise_for_status()
    records = response.json().get("parquet_files") or []
    result = [
        (str(record["url"]), str(record["filename"]))
        for record in records
        if record.get("split") == split
    ]
    if not result:
        raise RuntimeError(f"No converted parquet files for {dataset} split={split}")
    return result


def endpoint_url(url: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    parsed = urlsplit(url)
    return f"{endpoint}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")


def iter_split(
    dataset: str,
    split: str,
    *,
    cache_dir: str,
) -> Iterator[dict[str, Any]]:
    os.environ.setdefault("PARQUET_CACHE_DIR", cache_dir)
    token = os.environ.get("HF_TOKEN")
    for url, filename in converted_parquet_urls(dataset, split):
        local_path = download_parquet_to_local_cache(
            url=endpoint_url(url),
            repo_id=dataset,
            filename=f"refs-convert-parquet/default/{split}/{filename}",
            token=token,
        )
        rows = load_dataset(
            "parquet",
            data_files=[local_path],
            split="train",
            streaming=True,
        )
        for row in rows:
            if isinstance(row, dict):
                yield row


def reset_sqlite(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def guard_row(
    guard: CrossSourceDataGuard,
    row: dict[str, Any],
    source: str,
    exact_seen: set[int],
) -> str | None:
    content = joined_content(row)
    exact_key = stable_int(content.lower(), 0)
    if exact_key in exact_seen:
        return "exact_duplicate"
    if len(content) < 200:
        exact_seen.add(exact_key)
        return None
    decision, signature, segments = guard.evaluate(content)
    if not decision.keep:
        return decision.reason
    exact_seen.add(exact_key)
    guard.add(
        text=content,
        source=source,
        signature=signature,
        segments=segments,
    )
    return None


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    eval_output = Path(args.eval_output)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    index_path = Path(args.dedup_index)
    reset_sqlite(index_path)
    guard = CrossSourceDataGuard(
        index_path,
        similarity_threshold=0.82,
        max_duplicate_segment_fraction=0.30,
        contamination_path=args.contamination_path,
        contamination_ngram=13,
        contamination_lcs_threshold=0.60,
    )
    exact_seen: set[int] = set()
    rejects: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    eval_reservoir: list[dict[str, Any]] = []
    eval_accepted = 0
    rng = random.Random(args.seed)

    try:
        for raw in iter_split(
            args.sft_dataset,
            args.sft_eval_split,
            cache_dir=args.parquet_cache_dir,
        ):
            row = standardize_example(raw)
            if row is None:
                rejects["eval_invalid_conversation"] += 1
                continue
            reason = basic_quality_reason(row, tokenizer, args)
            if reason is None:
                reason = guard_row(guard, row, "smol-smoltalk-test", exact_seen)
            if reason is not None:
                rejects[f"eval_{reason}"] += 1
                continue
            eval_accepted += 1
            if len(eval_reservoir) < args.eval_size:
                eval_reservoir.append(row)
            else:
                replacement = rng.randrange(eval_accepted)
                if replacement < args.eval_size:
                    eval_reservoir[replacement] = row

        bucket_dir = output.with_suffix(output.suffix + ".buckets")
        shutil.rmtree(bucket_dir, ignore_errors=True)
        bucket_dir.mkdir(parents=True)
        bucket_count = 128
        handles = [
            (bucket_dir / f"{index:03d}.jsonl").open("w", encoding="utf-8")
            for index in range(bucket_count)
        ]
        train_rows = 0
        try:
            for seen, raw in enumerate(
                iter_split(
                    args.sft_dataset,
                    args.sft_split,
                    cache_dir=args.parquet_cache_dir,
                ),
                start=1,
            ):
                if seen > args.sft_source_limit:
                    break
                row = standardize_example(raw)
                if row is None:
                    rejects["train_invalid_conversation"] += 1
                    continue
                reason = basic_quality_reason(row, tokenizer, args)
                if reason is None:
                    reason = guard_row(
                        guard,
                        row,
                        f"smol-smoltalk-{row['source']}",
                        exact_seen,
                    )
                if reason is not None:
                    rejects[f"train_{reason}"] += 1
                    continue
                payload = json.dumps(row, ensure_ascii=False)
                key = stable_int(payload, args.seed)
                handles[key % bucket_count].write(f"{key:016x}\t{payload}\n")
                train_rows += 1
                source_counts[row["source"]] += 1
                if args.target_size > 0 and train_rows >= args.target_size:
                    break
                if seen % 10_000 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "sft_prepare_progress",
                                "seen": seen,
                                "accepted": train_rows,
                                "rejects": dict(rejects),
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
        finally:
            for handle in handles:
                handle.close()

        buckets = sorted(bucket_dir.glob("*.jsonl"))
        rng.shuffle(buckets)
        with output.open("w", encoding="utf-8") as destination:
            for bucket in buckets:
                lines = bucket.read_text(encoding="utf-8").splitlines()
                lines.sort()
                for line in lines:
                    destination.write(line.split("\t", 1)[1] + "\n")
        shutil.rmtree(bucket_dir)
        write_jsonl(eval_output, eval_reservoir)
    finally:
        guard.close()

    summary = {
        "sft_dataset": args.sft_dataset,
        "train_rows": train_rows,
        "eval_rows": len(eval_reservoir),
        "eval_candidates_after_filtering": eval_accepted,
        "source_counts": dict(source_counts),
        "rejects": dict(rejects),
        "max_example_tokens": args.max_example_tokens,
        "benchmark_contamination_ngram": 13,
        "benchmark_contamination_lcs_threshold": 0.60,
        "dedup_similarity_threshold": 0.82,
        "replay_rows": 0,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
