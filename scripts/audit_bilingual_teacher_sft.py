#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_boundary_protocol_tags(text: str) -> tuple[str, int]:
    cleaned, opening = re.subn(r"^\s*<request>\s*", "", text, count=1, flags=re.I)
    cleaned, closing = re.subn(r"\s*</request>\s*$", "", cleaned, count=1, flags=re.I)
    return cleaned.strip(), opening + closing


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed bilingual teacher shards.")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    partials = sorted(args.teacher_dir.glob("teacher-shard-*.partial"))
    if partials:
        raise ValueError(f"incomplete teacher shards remain: {partials}")
    teacher_paths = sorted(args.teacher_dir.glob("teacher-shard-*.jsonl"))
    rejection_paths = sorted(args.teacher_dir.glob("rejected-shard-*.jsonl"))
    if len(teacher_paths) != args.world_size or len(rejection_paths) != args.world_size:
        raise ValueError(
            f"expected {args.world_size} teacher/rejection shards, found "
            f"{len(teacher_paths)}/{len(rejection_paths)}"
        )

    jobs = read_jsonl(args.jobs)
    job_ids = [str(row["id"]) for row in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("teacher jobs contain duplicate ids")

    accepted: list[dict[str, Any]] = []
    for path in teacher_paths:
        accepted.extend(read_jsonl(path))
    rejected: list[dict[str, Any]] = []
    for path in rejection_paths:
        rejected.extend(read_jsonl(path))
    accepted_ids = [str(row["id"]) for row in accepted]
    rejected_ids = [str(row["id"]) for row in rejected]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("accepted teacher rows contain duplicate ids")
    if len(rejected_ids) != len(set(rejected_ids)):
        raise ValueError("rejected teacher rows contain duplicate ids")
    overlap = set(accepted_ids) & set(rejected_ids)
    if overlap:
        raise ValueError(f"accepted/rejected ids overlap: {len(overlap)}")
    observed_ids = set(accepted_ids) | set(rejected_ids)
    if observed_ids != set(job_ids):
        raise ValueError(
            f"teacher outputs do not partition jobs: missing={len(set(job_ids) - observed_ids)} "
            f"extra={len(observed_ids - set(job_ids))}"
        )

    source_counts: Counter[str] = Counter()
    protocol_leaks: Counter[str] = Counter()
    boundary_protocol_tags_cleaned = 0
    assistant_lengths: list[int] = []
    for row in accepted:
        messages = row.get("messages")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != [
            "user",
            "assistant",
        ]:
            raise ValueError(f"invalid message protocol for {row.get('id')}")
        user = str(messages[0].get("content") or "").strip()
        assistant = str(messages[1].get("content") or "").strip()
        user, cleaned = clean_boundary_protocol_tags(user)
        boundary_protocol_tags_cleaned += cleaned
        if len(re.findall(r"[\u4e00-\u9fff]", user)) < 4 or not assistant:
            raise ValueError(f"invalid bilingual content for {row.get('id')}")
        combined = f"{user}\n{assistant}".lower()
        for token in ("<request>", "</request>", "<think>", "</think>"):
            if token in combined:
                protocol_leaks[token] += 1
        source_counts[str(row.get("source") or "unknown")] += 1
        assistant_lengths.append(len(assistant))
    if protocol_leaks:
        raise ValueError(f"teacher protocol leaks detected: {dict(protocol_leaks)}")

    rejection_reasons = Counter(str(row.get("reason") or "unknown") for row in rejected)
    assistant_lengths.sort()
    payload = {
        "schema_version": 1,
        "status": (
            "complete_with_boundary_cleanup"
            if boundary_protocol_tags_cleaned
            else "complete"
        ),
        "jobs": len(jobs),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(jobs),
        "accepted_sources": dict(sorted(source_counts.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "boundary_protocol_tags_cleaned": boundary_protocol_tags_cleaned,
        "assistant_chars": {
            "min": assistant_lengths[0],
            "median": assistant_lengths[len(assistant_lengths) // 2],
            "p95": assistant_lengths[int(0.95 * (len(assistant_lengths) - 1))],
            "max": assistant_lengths[-1],
        },
        "jobs_sha256": sha256_file(args.jobs),
        "teacher_shards": [
            {"path": str(path), "rows": len(read_jsonl(path)), "sha256": sha256_file(path)}
            for path in teacher_paths
        ],
        "rejection_shards": [
            {"path": str(path), "rows": len(read_jsonl(path)), "sha256": sha256_file(path)}
            for path in rejection_paths
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
