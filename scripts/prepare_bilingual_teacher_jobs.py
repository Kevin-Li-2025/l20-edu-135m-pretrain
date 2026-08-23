#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUOTAS = {
    "smol-contraints": 2_500,
    "openhermes-50k": 2_500,
    "self-oss-instruct": 1_500,
    "smollm-rewrite-30k": 1_000,
    "smol-summarize-20k": 1_000,
    "explore-instruct-rewrite": 1_000,
    "smol-magpie-ultra-short": 500,
}


def selection_key(digest: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{digest}".encode()).digest(), "big")


def first_user_message(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ):
            return message["content"].strip()
    return None


def select_jobs(
    rows: Iterable[dict[str, Any]],
    quotas: dict[str, int],
    *,
    seed: int,
    max_prompt_chars: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    for row in rows:
        source = str(row.get("source") or "")
        if source not in quotas:
            continue
        digest = str(row.get("digest") or "")
        prompt = first_user_message(row)
        if not digest or not prompt:
            rejected[f"{source}:invalid"] += 1
            continue
        if len(prompt) > max_prompt_chars:
            rejected[f"{source}:too_long"] += 1
            continue
        candidates[source] += 1
        job = {
            "id": hashlib.sha256(f"zh:{digest}".encode()).hexdigest(),
            "source_digest": digest,
            "source": source,
            "prompt_en": prompt,
            "mode": "translate_and_answer_zh",
        }
        entry = (-selection_key(digest, seed), digest, job)
        heap = heaps[source]
        limit = quotas[source]
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    selected: list[dict[str, str]] = []
    selected_counts: Counter[str] = Counter()
    for source, limit in quotas.items():
        source_rows = [entry[2] for entry in heaps[source]]
        if len(source_rows) != limit:
            raise ValueError(
                f"source {source!r} has {len(source_rows)} valid rows, expected quota {limit}"
            )
        selected.extend(source_rows)
        selected_counts[source] = len(source_rows)
    selected.sort(key=lambda row: selection_key(row["id"], seed + 1))
    return selected, {
        "candidates": dict(sorted(candidates.items())),
        "selected": dict(sorted(selected_counts.items())),
        "rejected": dict(sorted(rejected.items())),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic bilingual teacher jobs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-prompt-chars", type=int, default=4_000)
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="SOURCE=COUNT",
        help="Override the default source quotas; specifying any overrides replaces the defaults.",
    )
    args = parser.parse_args()

    quotas = DEFAULT_QUOTAS
    if args.quota:
        quotas = {}
        for value in args.quota:
            source, count = value.rsplit("=", 1)
            quotas[source] = int(count)
    with args.input.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        jobs, stats = select_jobs(
            rows,
            quotas,
            seed=args.seed,
            max_prompt_chars=args.max_prompt_chars,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "quotas": quotas,
        "total": len(jobs),
        **stats,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", **manifest}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
