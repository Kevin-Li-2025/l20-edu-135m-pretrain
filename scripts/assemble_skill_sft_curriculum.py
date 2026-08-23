#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable, Iterator

from datasets import load_dataset

from l20_pretrain.contamination import normalize_tokens
from l20_pretrain.data_guard import BenchmarkContaminationIndex


DEFAULT_SMOLTALK_QUOTAS = {
    "smol-contraints": 33_631,
    "openhermes-50k": 25_000,
    "self-oss-instruct": 25_000,
    "smollm-rewrite-30k": 12_000,
    "smol-summarize-20k": 8_000,
    "explore-instruct-rewrite": 3_014,
    "everyday-conversations": 1_994,
    "smol-magpie-ultra-short": 15_000,
}


def selection_key(digest: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{digest}".encode()).digest(), "big")


def canonical_digest(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def row_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in row.get("messages", [])
        if isinstance(message, dict)
    )


def build_exact_prompt_index(
    heldout_path: Path, chat_paths: Iterable[Path]
) -> dict[tuple[str, ...], str]:
    index: dict[tuple[str, ...], str] = {}
    with heldout_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            tokens = tuple(normalize_tokens(str(raw.get("text") or "")))
            if tokens:
                index.setdefault(tokens, str(raw.get("benchmark") or "heldout"))
    for path in chat_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                for message in raw.get("messages", []):
                    if message.get("role") != "user":
                        continue
                    tokens = tuple(normalize_tokens(str(message.get("content") or "")))
                    if tokens:
                        index.setdefault(tokens, f"chat:{path.stem}")
    return index


def exact_prompt_match(
    row: dict[str, Any], index: dict[tuple[str, ...], str]
) -> str | None:
    for message in row.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        tokens = tuple(normalize_tokens(str(message.get("content") or "")))
        if tokens in index:
            return index[tokens]
    return None


def select_smoltalk_rows(
    rows: Iterable[dict[str, Any]], quotas: dict[str, int], *, seed: int
) -> list[dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source") or "")
        if source not in quotas:
            continue
        digest = str(row.get("digest") or "")
        if not digest:
            continue
        entry = (-selection_key(digest, seed), digest, row)
        heap = heaps[source]
        if len(heap) < quotas[source]:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected: list[dict[str, Any]] = []
    for source, quota in quotas.items():
        source_rows = [entry[2] for entry in heaps[source]]
        if len(source_rows) != quota:
            raise ValueError(f"source {source!r} selected {len(source_rows)}, expected {quota}")
        selected.extend(source_rows)
    return selected


def gsm8k_rows(dataset: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for raw in dataset:
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            continue
        answer = re.sub(r"<<[^<>]*>>", "", answer)
        answer = answer.replace("#### ", "Final answer: ")
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        yield {
            "messages": messages,
            "source": "gsm8k-train",
            "digest": canonical_digest(messages),
        }


def arithmetic_rows(count: int, *, seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[tuple[str, int, int, str]] = set()
    operations = ("add", "subtract", "multiply", "divide")
    while len(seen) < count:
        operation = operations[len(seen) % len(operations)]
        language = "zh" if (len(seen) // len(operations)) % 2 else "en"
        if operation == "add":
            left, right = rng.randint(2, 999), rng.randint(2, 999)
            answer = left + right
            prompt = (
                f"计算 {left} 加 {right}。只输出数字。"
                if language == "zh"
                else f"What is {left} plus {right}? Output only the number."
            )
        elif operation == "subtract":
            right = rng.randint(2, 500)
            answer = rng.randint(2, 500)
            left = right + answer
            prompt = (
                f"计算 {left} 减 {right}。只输出数字。"
                if language == "zh"
                else f"What is {left} minus {right}? Output only the number."
            )
        elif operation == "multiply":
            left, right = rng.randint(2, 30), rng.randint(2, 30)
            answer = left * right
            prompt = (
                f"计算 {left} 乘以 {right}。只输出数字。"
                if language == "zh"
                else f"What is {left} multiplied by {right}? Output only the number."
            )
        else:
            right, answer = rng.randint(2, 30), rng.randint(2, 30)
            left = right * answer
            prompt = (
                f"计算 {left} 除以 {right}。只输出数字。"
                if language == "zh"
                else f"What is {left} divided by {right}? Output only the number."
            )
        key = (operation, left, right, language)
        if key in seen:
            continue
        seen.add(key)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(answer)},
        ]
        yield {
            "messages": messages,
            "source": f"synthetic-arithmetic-{language}",
            "digest": canonical_digest(messages),
        }


def clean_boundary_protocol_tags(text: str) -> tuple[str, int]:
    cleaned, opening = re.subn(r"^\s*<request>\s*", "", text, count=1, flags=re.I)
    cleaned, closing = re.subn(r"\s*</request>\s*$", "", cleaned, count=1, flags=re.I)
    return cleaned.strip(), opening + closing


def teacher_rows(
    paths: Iterable[Path], *, stats: Counter[str] | None = None
) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                messages = raw.get("messages")
                if not isinstance(messages, list):
                    continue
                normalized = [
                    {"role": str(message["role"]), "content": str(message["content"]).strip()}
                    for message in messages
                    if isinstance(message, dict)
                    and message.get("role") in {"system", "user", "assistant"}
                    and str(message.get("content") or "").strip()
                ]
                if len(normalized) < 2 or normalized[-1]["role"] != "assistant":
                    continue
                boundary_cleanups = 0
                for message in normalized:
                    if message["role"] != "user":
                        continue
                    message["content"], count = clean_boundary_protocol_tags(
                        message["content"]
                    )
                    boundary_cleanups += count
                combined = "\n".join(message["content"].lower() for message in normalized)
                if any(
                    token in combined
                    for token in ("<request>", "</request>", "<think>", "</think>")
                ):
                    raise ValueError(f"embedded teacher protocol token in {path}")
                if stats is not None:
                    stats["boundary_protocol_tags_cleaned"] += boundary_cleanups
                yield {
                    "messages": normalized,
                    "source": str(raw.get("source") or "qwen3-8b-zh"),
                    "digest": canonical_digest(normalized),
                }


def chat_contamination_index(paths: Iterable[Path]) -> tuple[BenchmarkContaminationIndex, Any]:
    temp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                text = " ".join(
                    message["content"]
                    for message in raw.get("messages", [])
                    if message.get("role") == "user"
                )
                temp.write(
                    json.dumps({"benchmark": f"chat:{path.stem}", "text": text}) + "\n"
                )
    temp.close()
    return BenchmarkContaminationIndex(temp.name), temp.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a balanced skill SFT curriculum.")
    parser.add_argument("--smoltalk", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--teacher-recipe", type=Path, required=True)
    parser.add_argument("--heldout-index", type=Path, required=True)
    parser.add_argument("--chat-prompts", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gsm8k-revision", required=True)
    parser.add_argument("--arithmetic-examples", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    with args.smoltalk.open(encoding="utf-8") as handle:
        smoltalk = select_smoltalk_rows(
            (json.loads(line) for line in handle if line.strip()),
            DEFAULT_SMOLTALK_QUOTAS,
            seed=args.seed,
        )
    gsm = list(
        gsm8k_rows(
            load_dataset(
                "openai/gsm8k",
                "main",
                split="train",
                revision=args.gsm8k_revision,
            )
        )
    )
    arithmetic = list(arithmetic_rows(args.arithmetic_examples, seed=args.seed + 2))
    teacher_paths = sorted(args.teacher_dir.glob("teacher-shard-*.jsonl"))
    if not teacher_paths:
        raise ValueError(f"no complete teacher shards found below {args.teacher_dir}")
    teacher_stats: Counter[str] = Counter()
    teacher = list(teacher_rows(teacher_paths, stats=teacher_stats))

    indexes = [BenchmarkContaminationIndex(args.heldout_index)]
    chat_index, temp_path = chat_contamination_index(args.chat_prompts)
    indexes.append(chat_index)
    exact_index = build_exact_prompt_index(args.heldout_index, args.chat_prompts)
    candidates = [*smoltalk, *gsm, *arithmetic, *teacher]
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    try:
        for row in candidates:
            digest = str(row["digest"])
            if digest in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(digest)
            exact_match = exact_prompt_match(row, exact_index)
            if exact_match is not None:
                rejected[f"contamination-exact:{exact_match}"] += 1
                continue
            match = next(
                (match for index in indexes if (match := index.match(row_text(row))) is not None),
                None,
            )
            if match is not None:
                rejected[f"contamination:{match[0]}"] += 1
                continue
            accepted.append(row)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    accepted.sort(key=lambda row: selection_key(row["digest"], args.seed + 3))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_counts = Counter(row["source"] for row in accepted)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "smoltalk": str(args.smoltalk),
        "smoltalk_sha256": sha256_file(args.smoltalk),
        "smoltalk_quotas": DEFAULT_SMOLTALK_QUOTAS,
        "teacher_shards": [
            {"path": str(path), "sha256": sha256_file(path)} for path in teacher_paths
        ],
        "teacher_processing": dict(sorted(teacher_stats.items())),
        "teacher_recipe": {
            "path": str(args.teacher_recipe),
            "sha256": sha256_file(args.teacher_recipe),
        },
        "gsm8k": {"dataset": "openai/gsm8k", "revision": args.gsm8k_revision},
        "arithmetic_examples_requested": args.arithmetic_examples,
        "candidates": len(candidates),
        "selected": len(accepted),
        "selected_sources": dict(sorted(source_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "heldout_index": str(args.heldout_index),
        "heldout_index_sha256": sha256_file(args.heldout_index),
        "chat_prompts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.chat_prompts
        ],
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "assembler_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **manifest}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
