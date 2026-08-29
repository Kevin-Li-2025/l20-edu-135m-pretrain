#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi

from l20_pretrain.data_guard import BenchmarkContaminationIndex


ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class PreparedRow:
    digest: str
    messages: list[dict[str, str]]
    source: str

    def payload(self) -> dict[str, Any]:
        return {"messages": self.messages, "source": self.source, "digest": self.digest}


def canonical_messages(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_row(row: Any, *, max_chars: int) -> tuple[PreparedRow | None, str]:
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return None, "invalid_messages"

    messages: list[dict[str, str]] = []
    for item in row["messages"]:
        if not isinstance(item, dict):
            return None, "invalid_message"
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None, "invalid_message"
        role = role.strip().lower()
        content = content.strip()
        if role not in ALLOWED_ROLES or not content:
            return None, "invalid_message"
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None, "too_few_messages"
    if messages[0]["role"] == "system":
        conversation = messages[1:]
        if not conversation or any(message["role"] == "system" for message in conversation):
            return None, "invalid_system_position"
    else:
        conversation = messages
    if conversation[0]["role"] != "user" or conversation[-1]["role"] != "assistant":
        return None, "invalid_conversation_boundary"
    for index, message in enumerate(conversation):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            return None, "non_alternating_roles"

    canonical = canonical_messages(messages)
    if len(canonical) > max_chars:
        return None, "too_long"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    source = str(row.get("source") or "unknown").strip() or "unknown"
    return PreparedRow(digest=digest, messages=messages, source=source), "ok"


def selection_key(digest: str, seed: int) -> int:
    payload = f"{seed}:{digest}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_smallest(rows: Iterable[PreparedRow], limit: int | None, seed: int) -> list[PreparedRow]:
    if limit is None:
        return list(rows)
    heap: list[tuple[int, str, PreparedRow]] = []
    for row in rows:
        key = selection_key(row.digest, seed)
        entry = (-key, row.digest, row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def prepare_split(
    rows: Iterable[Any],
    *,
    max_chars: int,
    max_examples: int | None,
    seed: int,
    contamination: BenchmarkContaminationIndex,
    forbidden_digests: set[str] | None = None,
) -> tuple[list[PreparedRow], dict[str, Any]]:
    rejected: Counter[str] = Counter()
    source_candidates: Counter[str] = Counter()
    seen = set(forbidden_digests or ())

    def candidates() -> Iterator[PreparedRow]:
        for raw in rows:
            row, reason = normalize_row(raw, max_chars=max_chars)
            if row is None:
                rejected[reason] += 1
                continue
            if row.digest in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(row.digest)
            text = "\n".join(message["content"] for message in row.messages)
            match = contamination.match(text)
            if match is not None:
                rejected[f"contamination:{match[0]}"] += 1
                continue
            source_candidates[row.source] += 1
            yield row

    selected = select_smallest(candidates(), max_examples, seed)
    source_selected = Counter(row.source for row in selected)
    return selected, {
        "selected": len(selected),
        "candidate_sources": dict(sorted(source_candidates.items())),
        "selected_sources": dict(sorted(source_selected.items())),
        "rejected": dict(sorted(rejected.items())),
    }


def write_jsonl(path: Path, rows: Iterable[PreparedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.payload(), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int_or_none(value: str) -> int | None:
    if value.lower() in {"none", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, 'all', or 'none'")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic, clean SFT JSONL files.")
    parser.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--contamination-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train", type=positive_int_or_none, default=50_000)
    parser.add_argument("--max-eval", type=positive_int_or_none, default=2_000)
    parser.add_argument("--max-chars", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    contamination = BenchmarkContaminationIndex(args.contamination_index)
    eval_rows, eval_stats = prepare_split(
        load_dataset(args.dataset, split="test", revision=revision),
        max_chars=args.max_chars,
        max_examples=args.max_eval,
        seed=args.seed + 1,
        contamination=contamination,
    )
    train_rows, train_stats = prepare_split(
        load_dataset(args.dataset, split="train", revision=revision),
        max_chars=args.max_chars,
        max_examples=args.max_train,
        seed=args.seed,
        contamination=contamination,
        forbidden_digests={row.digest for row in eval_rows},
    )

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "revision": revision,
        "seed": args.seed,
        "max_chars": args.max_chars,
        "contamination_index": str(args.contamination_index),
        "contamination_index_sha256": sha256_file(args.contamination_index),
        "train": {**train_stats, "path": str(train_path), "sha256": sha256_file(train_path)},
        "eval": {**eval_stats, "path": str(eval_path), "sha256": sha256_file(eval_path)},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", "manifest": str(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
