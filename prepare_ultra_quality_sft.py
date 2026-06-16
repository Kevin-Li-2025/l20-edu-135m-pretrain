#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any


BAD_PATTERNS = [
    "as an ai language model",
    "i cannot browse",
    "i don't have access to",
    "i do not have access to",
    "sorry, but i can't",
    "i'm sorry, but i can't",
]

SOURCE_PRIORS = {
    "openhermes-50k": 1.15,
    "self-oss-instruct": 1.08,
    "explore-instruct-rewrite": 1.08,
    "smol-contraints": 1.05,
    "everyday-conversations": 1.02,
    "smollm-rewrite-30k": 1.00,
    "smol-summarize-20k": 0.98,
    "longalign": 0.96,
    "smol-magpie-ultra-short": 0.62,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-pass ultra-quality SFT selector.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--target-size", type=int, default=30000)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--min-assistant-chars", type=int, default=220)
    parser.add_argument("--max-example-chars", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def clean_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None
        out.append({"role": role, "content": content})
    if not out:
        return None
    conv = out[1:] if out[0]["role"] == "system" else out
    if not conv or conv[0]["role"] != "user" or conv[-1]["role"] != "assistant":
        return None
    expected = "user"
    for msg in conv:
        if msg["role"] != expected:
            return None
        expected = "assistant" if expected == "user" else "user"
    return out


def all_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(m["content"] for m in messages)


def assistant_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(m["content"] for m in messages if m["role"] == "assistant")


def unique_word_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def repeated_line_fraction(text: str) -> float:
    lines = [x.strip().lower() for x in text.splitlines() if x.strip()]
    if len(lines) < 4:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def classify(messages: list[dict[str, str]], source: str) -> str:
    text = all_text(messages).lower()
    if any(k in text for k in ["python", "javascript", "typescript", "sql", "code", "function", "debug"]):
        return "coding"
    if any(k in text for k in ["prove", "solve", "reason", "step by step", "why", "logic", "math"]):
        return "reasoning"
    if any(k in text for k in ["summarize", "rewrite", "write", "essay", "story", "email", "tone"]):
        return "writing"
    if any(k in text for k in ["format", "json", "table", "bullet", "csv", "schema"]):
        return "format"
    if any(k in text for k in ["explain", "what is", "compare", "history", "science", "concept"]):
        return "knowledge"
    if "summarize" in source:
        return "summarization"
    if "constraints" in source or "contraints" in source:
        return "format"
    return "general"


def quality_score(row: dict[str, Any], messages: list[dict[str, str]], args: argparse.Namespace) -> tuple[float, str | None, str]:
    text = all_text(messages)
    answer = assistant_text(messages)
    lower = answer.lower()
    source = str(row.get("source") or row.get("dataset") or "unknown")
    category = classify(messages, source)
    if len(text) > args.max_example_chars:
        return 0.0, "too_long", category
    if len(answer) < args.min_assistant_chars:
        return 0.0, "assistant_too_short", category
    if any(p in lower for p in BAD_PATTERNS):
        return 0.0, "bad_disclaimer", category
    if repeated_line_fraction(answer) > 0.20:
        return 0.0, "repeated_lines", category
    uwr = unique_word_ratio(answer)
    if uwr < 0.18:
        return 0.0, "low_unique_words", category
    turns = sum(1 for m in messages if m["role"] == "assistant")
    has_structure = bool(re.search(r"(^|\n)([-*]|\d+\.|#{1,3})\s+", answer)) or "```" in answer
    length_bonus = min(1.0, math.log1p(len(answer)) / math.log(3500))
    structure_bonus = 0.16 if has_structure else 0.0
    turn_bonus = min(0.18, 0.05 * max(0, turns - 1))
    category_bonus = {
        "reasoning": 0.20,
        "knowledge": 0.16,
        "coding": 0.12,
        "format": 0.12,
        "writing": 0.10,
        "summarization": 0.08,
        "general": 0.04,
    }.get(category, 0.0)
    prior = SOURCE_PRIORS.get(source, 1.0)
    score = prior * (0.50 + 0.45 * length_bonus + structure_bonus + turn_bonus + category_bonus)
    return score, None, category


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    buckets: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    rejects = Counter()
    seen = set()
    source_counts = Counter()
    category_counts = Counter()
    for input_path in args.inputs:
        for row in iter_jsonl(Path(input_path)):
            messages = clean_messages(row.get("messages"))
            if messages is None:
                rejects["invalid_messages"] += 1
                continue
            normalized = re.sub(r"\s+", " ", all_text(messages).lower())
            key = stable_hash(normalized)
            if key in seen:
                rejects["exact_duplicate"] += 1
                continue
            seen.add(key)
            score, reason, category = quality_score(row, messages, args)
            if reason:
                rejects[reason] += 1
                continue
            out = {"messages": messages, "source": str(row.get("source") or "unknown"), "quality_score": round(score, 6), "category": category}
            buckets[category].append((score, stable_hash(json.dumps(out, sort_keys=True)), out))
            source_counts[out["source"]] += 1
            category_counts[category] += 1

    quotas = {
        "reasoning": 0.24,
        "knowledge": 0.18,
        "writing": 0.16,
        "format": 0.14,
        "coding": 0.10,
        "summarization": 0.08,
        "general": 0.10,
    }
    selected: list[dict[str, Any]] = []
    selected_counts = Counter()
    leftovers: list[tuple[float, int, dict[str, Any]]] = []
    for category, rows in buckets.items():
        rows.sort(key=lambda x: (-x[0], x[1]))
        quota = int(args.target_size * quotas.get(category, 0.05))
        take = rows[:quota]
        selected.extend(row for _, _, row in take)
        selected_counts[category] += len(take)
        leftovers.extend(rows[quota:])
    leftovers.sort(key=lambda x: (-x[0], x[1]))
    for _, _, row in leftovers:
        if len(selected) >= args.target_size:
            break
        selected.append(row)
        selected_counts[row["category"]] += 1
    rng.shuffle(selected)
    eval_rows = selected[: args.eval_size]
    train_rows = selected[args.eval_size : args.eval_size + args.target_size]
    write_jsonl(Path(args.output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)
    summary = {
        "inputs": args.inputs,
        "target_size": args.target_size,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "rejects": dict(rejects),
        "candidate_categories": dict(category_counts),
        "selected_categories": dict(selected_counts),
        "source_counts_seen_after_filter": dict(source_counts.most_common(30)),
        "mean_quality_score": sum(r["quality_score"] for r in train_rows) / max(1, len(train_rows)),
    }
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
