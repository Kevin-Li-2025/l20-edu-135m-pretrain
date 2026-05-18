from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class ContaminationReport:
    train_ngrams: int
    benchmark_ngrams: int
    overlap_ngrams: int
    overlap_rate: float
    status: str


def normalize_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def word_ngrams(text: str, n: int) -> set[str]:
    tokens = normalize_tokens(text)
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def iter_strings(payload: Any) -> Iterator[str]:
    if isinstance(payload, str):
        if payload.strip():
            yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from iter_strings(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            yield from iter_strings(value)


def iter_texts_from_file(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in text.split("\n\n"):
            chunk = chunk.strip()
            if chunk:
                yield chunk
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield from iter_strings(payload)
        return

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return
        yield from iter_strings(payload)


def iter_texts_from_path(path: str | Path) -> Iterator[str]:
    path = Path(path)
    if path.is_file():
        yield from iter_texts_from_file(path)
    else:
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".json", ".jsonl"}:
                yield from iter_texts_from_file(file_path)


def collect_ngrams(texts: Iterable[str], n: int, limit: int | None = None) -> set[str]:
    output: set[str] = set()
    for index, text in enumerate(texts):
        if limit is not None and index >= limit:
            break
        output.update(word_ngrams(text, n))
    return output


def contamination_report(
    train_path: str | Path,
    benchmark_path: str | Path,
    *,
    ngram: int = 13,
    max_train_texts: int | None = None,
    max_benchmark_texts: int | None = None,
    fail_threshold: float = 0.001,
) -> ContaminationReport:
    train_ngrams = collect_ngrams(iter_texts_from_path(train_path), ngram, max_train_texts)
    benchmark_ngrams = collect_ngrams(
        iter_texts_from_path(benchmark_path),
        ngram,
        max_benchmark_texts,
    )
    overlap = train_ngrams & benchmark_ngrams
    overlap_rate = len(overlap) / max(1, len(benchmark_ngrams))
    status = "pass" if overlap_rate <= fail_threshold else "fail"
    return ContaminationReport(
        train_ngrams=len(train_ngrams),
        benchmark_ngrams=len(benchmark_ngrams),
        overlap_ngrams=len(overlap),
        overlap_rate=overlap_rate,
        status=status,
    )
