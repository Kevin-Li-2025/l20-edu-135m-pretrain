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
class PreferenceRow:
    digest: str
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
    score_chosen: float | None
    score_rejected: float | None

    def payload(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "score_chosen": self.score_chosen,
            "score_rejected": self.score_rejected,
            "digest": self.digest,
        }


def normalize_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        role = role.strip().lower()
        content = content.strip()
        if role not in ALLOWED_ROLES or not content:
            return None
        messages.append({"role": role, "content": content})
    return messages or None


def optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_preference(
    row: Any,
    *,
    max_prompt_chars: int,
    max_completion_chars: int,
) -> tuple[PreferenceRow | None, str]:
    if not isinstance(row, dict):
        return None, "invalid_row"
    chosen = normalize_messages(row.get("chosen"))
    rejected = normalize_messages(row.get("rejected"))
    if not chosen or not rejected:
        return None, "invalid_messages"
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        return None, "missing_assistant_completion"
    if chosen[:-1] != rejected[:-1] or not chosen[:-1]:
        return None, "prompt_mismatch"
    prompt = chosen[:-1]
    if prompt[-1]["role"] != "user":
        return None, "invalid_prompt_boundary"

    prompt_text = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    chosen_text = chosen[-1]["content"]
    rejected_text = rejected[-1]["content"]
    if len(prompt_text) > max_prompt_chars:
        return None, "prompt_too_long"
    if max(len(chosen_text), len(rejected_text)) > max_completion_chars:
        return None, "completion_too_long"
    if chosen_text == rejected_text:
        return None, "identical_completions"

    canonical = json.dumps(
        {"prompt": prompt, "chosen": chosen[-1:], "rejected": rejected[-1:]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        PreferenceRow(
            digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            prompt=prompt,
            chosen=chosen[-1:],
            rejected=rejected[-1:],
            score_chosen=optional_float(row.get("score_chosen")),
            score_rejected=optional_float(row.get("score_rejected")),
        ),
        "ok",
    )


def selection_key(digest: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{digest}".encode()).digest(), "big")


def select_smallest(
    rows: Iterable[PreferenceRow], limit: int | None, seed: int
) -> list[PreferenceRow]:
    if limit is None:
        return list(rows)
    heap: list[tuple[int, str, PreferenceRow]] = []
    for row in rows:
        entry = (-selection_key(row.digest, seed), row.digest, row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def prepare_split(
    rows: Iterable[Any],
    *,
    max_prompt_chars: int,
    max_completion_chars: int,
    max_examples: int | None,
    seed: int,
    contamination: BenchmarkContaminationIndex,
    forbidden_digests: set[str] | None = None,
) -> tuple[list[PreferenceRow], dict[str, Any]]:
    rejected_counts: Counter[str] = Counter()
    seen = set(forbidden_digests or ())

    def candidates() -> Iterator[PreferenceRow]:
        for raw in rows:
            row, reason = normalize_preference(
                raw,
                max_prompt_chars=max_prompt_chars,
                max_completion_chars=max_completion_chars,
            )
            if row is None:
                rejected_counts[reason] += 1
                continue
            if row.digest in seen:
                rejected_counts["duplicate"] += 1
                continue
            seen.add(row.digest)
            prompt_text = "\n".join(message["content"] for message in row.prompt)
            match = contamination.match(prompt_text)
            if match is not None:
                rejected_counts[f"contamination:{match[0]}"] += 1
                continue
            candidates_seen[0] += 1
            yield row

    candidates_seen = [0]
    selected = select_smallest(candidates(), max_examples, seed)
    score_gaps = [
        row.score_chosen - row.score_rejected
        for row in selected
        if row.score_chosen is not None and row.score_rejected is not None
    ]
    return selected, {
        "candidates": candidates_seen[0],
        "selected": len(selected),
        "rejected": dict(sorted(rejected_counts.items())),
        "mean_score_gap": sum(score_gaps) / len(score_gaps) if score_gaps else None,
    }


def write_jsonl(path: Path, rows: Iterable[PreferenceRow]) -> None:
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
    parser = argparse.ArgumentParser(description="Prepare clean preference JSONL data.")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--contamination-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train", type=positive_int_or_none, default=None)
    parser.add_argument("--max-eval", type=positive_int_or_none, default=2_000)
    parser.add_argument("--max-prompt-chars", type=int, default=8_000)
    parser.add_argument("--max-completion-chars", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    contamination = BenchmarkContaminationIndex(args.contamination_index)
    eval_rows, eval_stats = prepare_split(
        load_dataset(args.dataset, split="test_prefs", revision=revision),
        max_prompt_chars=args.max_prompt_chars,
        max_completion_chars=args.max_completion_chars,
        max_examples=args.max_eval,
        seed=args.seed + 1,
        contamination=contamination,
    )
    train_rows, train_stats = prepare_split(
        load_dataset(args.dataset, split="train_prefs", revision=revision),
        max_prompt_chars=args.max_prompt_chars,
        max_completion_chars=args.max_completion_chars,
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
        "max_prompt_chars": args.max_prompt_chars,
        "max_completion_chars": args.max_completion_chars,
        "contamination_index": str(args.contamination_index),
        "contamination_index_sha256": sha256_file(args.contamination_index),
        "train": {**train_stats, "path": str(train_path), "sha256": sha256_file(train_path)},
        "eval": {**eval_stats, "path": str(eval_path), "sha256": sha256_file(eval_path)},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "manifest": str(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
