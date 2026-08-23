from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import struct
from typing import Iterable

import numpy as np
import xxhash

from .contamination import normalize_tokens
from .quality import stable_hash


PARAGRAPH_RE = re.compile(r"\n\s*\n+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
SPACE_RE = re.compile(r"\s+")
MASK64 = (1 << 64) - 1
MINHASH_SEEDS = np.asarray(
    [
        xxhash.xxh64_intdigest(f"l20-minhash-{index}".encode("utf-8"))
        for index in range(64)
    ],
    dtype=np.uint64,
)


@dataclass(frozen=True)
class GuardDecision:
    keep: bool
    text: str
    reason: str = "ok"
    duplicate_fraction: float = 0.0
    benchmark: str | None = None
    lcs_ratio: float = 0.0


def canonical_segment(text: str) -> str:
    text = URL_RE.sub("<url>", text.lower())
    text = EMAIL_RE.sub("<email>", text)
    text = NUMBER_RE.sub("<num>", text)
    return SPACE_RE.sub(" ", text).strip()


def segment_hash(text: str) -> str:
    return stable_hash(canonical_segment(text))


def word_shingle_hashes(text: str, n: int = 5, max_shingles: int = 4096) -> np.ndarray:
    tokens = normalize_tokens(text)
    count = len(tokens) - n + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    hashes = {
        xxhash.xxh64_intdigest(
            " ".join(tokens[index : index + n]).encode("utf-8")
        )
        for index in range(count)
    }
    values = np.fromiter(hashes, dtype=np.uint64)
    if values.size <= max_shingles:
        return values
    return np.partition(values, max_shingles - 1)[:max_shingles]


def minhash_signature(text: str) -> tuple[int, ...]:
    hashes = word_shingle_hashes(text)
    if hashes.size == 0:
        return ()
    signature: list[int] = []
    multiplier = np.uint64(0x9E3779B185EBCA87)
    for seed in MINHASH_SEEDS:
        mixed = (hashes ^ seed) * multiplier
        signature.append(int(mixed.min()))
    return tuple(signature)


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def signature_bands(signature: tuple[int, ...], rows: int = 4) -> Iterable[tuple[int, str]]:
    for band in range(0, len(signature), rows):
        values = signature[band : band + rows]
        payload = struct.pack(f"<{len(values)}Q", *values)
        yield band // rows, xxhash.xxh64_hexdigest(payload)


def packed_signature(signature: tuple[int, ...]) -> bytes:
    return struct.pack(f"<{len(signature)}Q", *signature)


def unpacked_signature(payload: bytes) -> tuple[int, ...]:
    if not payload:
        return ()
    return struct.unpack(f"<{len(payload) // 8}Q", payload)


def token_lcs_ratio(document: list[str], benchmark: list[str]) -> float:
    if not document or not benchmark:
        return 0.0
    previous = [0] * (len(benchmark) + 1)
    for token in document:
        current = [0]
        for index, benchmark_token in enumerate(benchmark, start=1):
            if token == benchmark_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
        if previous[-1] == len(benchmark):
            return 1.0
    return previous[-1] / len(benchmark)


class BenchmarkContaminationIndex:
    def __init__(self, path: str | Path, *, ngram: int = 13, lcs_threshold: float = 0.6):
        self.path = Path(path)
        self.ngram = ngram
        self.lcs_threshold = lcs_threshold
        self.records: list[tuple[str, list[str]]] = []
        self.ngrams: dict[int, list[int]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Benchmark contamination file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                tokens = normalize_tokens(str(payload["text"]))
                if len(tokens) < self.ngram:
                    continue
                record_id = len(self.records)
                self.records.append((str(payload["benchmark"]), tokens))
                for index in range(len(tokens) - self.ngram + 1):
                    digest = xxhash.xxh64_intdigest(
                        " ".join(tokens[index : index + self.ngram]).encode("utf-8")
                    )
                    self.ngrams[digest].append(record_id)

    def match(self, text: str) -> tuple[str, float] | None:
        tokens = normalize_tokens(text)
        if len(tokens) < self.ngram:
            return None
        votes: dict[int, int] = defaultdict(int)
        for index in range(len(tokens) - self.ngram + 1):
            digest = xxhash.xxh64_intdigest(
                " ".join(tokens[index : index + self.ngram]).encode("utf-8")
            )
            for record_id in self.ngrams.get(digest, ()):
                votes[record_id] += 1
        for record_id, _ in sorted(votes.items(), key=lambda item: item[1], reverse=True)[:32]:
            benchmark, benchmark_tokens = self.records[record_id]
            ratio = token_lcs_ratio(tokens, benchmark_tokens)
            if ratio >= self.lcs_threshold:
                return benchmark, ratio
        return None


class CrossSourceDataGuard:
    def __init__(
        self,
        index_path: str | Path,
        *,
        similarity_threshold: float = 0.82,
        max_duplicate_segment_fraction: float = 0.30,
        contamination_path: str | Path | None = None,
        contamination_ngram: int = 13,
        contamination_lcs_threshold: float = 0.6,
    ):
        self.path = Path(index_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.max_duplicate_segment_fraction = max_duplicate_segment_fraction
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-16384")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA wal_autocheckpoint=4096")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                digest TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                signature BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bands (
                band INTEGER NOT NULL,
                bucket TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY (band, bucket, digest)
            );
            CREATE INDEX IF NOT EXISTS bands_lookup ON bands(band, bucket);
            CREATE TABLE IF NOT EXISTS segments (
                kind TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY (kind, digest)
            );
            """
        )
        self.contamination = (
            BenchmarkContaminationIndex(
                contamination_path,
                ngram=contamination_ngram,
                lcs_threshold=contamination_lcs_threshold,
            )
            if contamination_path
            else None
        )
        self.pending = 0

    def close(self) -> None:
        self.checkpoint(truncate=True)
        self.connection.close()

    def checkpoint(self, *, truncate: bool = False) -> None:
        self.connection.commit()
        self.pending = 0
        mode = "TRUNCATE" if truncate else "PASSIVE"
        self.connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()

    def _segment_exists(self, kind: str, digest: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM segments WHERE kind = ? AND digest = ? LIMIT 1",
            (kind, digest),
        ).fetchone()
        return row is not None

    def clean_segments(self, text: str) -> tuple[str, set[tuple[str, str]], float]:
        paragraphs = [part.strip() for part in PARAGRAPH_RE.split(text) if part.strip()]
        output: list[str] = []
        new_segments: set[tuple[str, str]] = set()
        local_segments: set[tuple[str, str]] = set()
        considered = 0
        duplicates = 0

        for paragraph in paragraphs:
            paragraph_key = ("paragraph", segment_hash(paragraph))
            paragraph_trackable = len(paragraph) >= 240
            if paragraph_trackable:
                considered += 1
                if paragraph_key in local_segments or self._segment_exists(*paragraph_key):
                    duplicates += 1
                    continue
                local_segments.add(paragraph_key)
                new_segments.add(paragraph_key)

            sentences = [part.strip() for part in SENTENCE_RE.split(paragraph) if part.strip()]
            kept_sentences: list[str] = []
            for sentence in sentences:
                sentence_key = ("sentence", segment_hash(sentence))
                sentence_trackable = len(sentence) >= 160 and len(normalize_tokens(sentence)) >= 16
                if sentence_trackable:
                    considered += 1
                    if sentence_key in local_segments or self._segment_exists(*sentence_key):
                        duplicates += 1
                        continue
                    local_segments.add(sentence_key)
                    new_segments.add(sentence_key)
                kept_sentences.append(sentence)
            if kept_sentences:
                output.append(" ".join(kept_sentences))

        fraction = duplicates / max(1, considered)
        return "\n\n".join(output), new_segments, fraction

    def _near_duplicate(self, signature: tuple[int, ...]) -> bool:
        bands = list(signature_bands(signature))
        if not bands:
            return False
        conditions = " OR ".join("(band = ? AND bucket = ?)" for _ in bands)
        parameters = [value for pair in bands for value in pair]
        rows = self.connection.execute(
            f"""
            SELECT digest, COUNT(*) AS band_hits
            FROM bands
            WHERE {conditions}
            GROUP BY digest
            ORDER BY band_hits DESC
            LIMIT 4096
            """,
            parameters,
        )
        for digest, _ in rows:
            row = self.connection.execute(
                "SELECT signature FROM documents WHERE digest = ?", (digest,)
            ).fetchone()
            if row and signature_similarity(signature, unpacked_signature(row[0])) >= self.similarity_threshold:
                return True
        return False

    def evaluate(self, text: str) -> tuple[GuardDecision, tuple[int, ...], set[tuple[str, str]]]:
        cleaned, segments, duplicate_fraction = self.clean_segments(text)
        if duplicate_fraction >= self.max_duplicate_segment_fraction:
            return (
                GuardDecision(
                    False,
                    cleaned,
                    "segment_duplicate",
                    duplicate_fraction=duplicate_fraction,
                ),
                (),
                set(),
            )
        if len(cleaned) < 200:
            return GuardDecision(False, cleaned, "segment_cleanup_too_short"), (), set()

        signature = minhash_signature(cleaned)
        if self._near_duplicate(signature):
            return GuardDecision(False, cleaned, "near_duplicate"), (), set()

        if self.contamination:
            match = self.contamination.match(cleaned)
            if match:
                benchmark, ratio = match
                return (
                    GuardDecision(
                        False,
                        cleaned,
                        "benchmark_13gram_lcs",
                        benchmark=benchmark,
                        lcs_ratio=ratio,
                    ),
                    (),
                    set(),
                )
        return GuardDecision(True, cleaned, duplicate_fraction=duplicate_fraction), signature, segments

    def add(
        self,
        *,
        text: str,
        source: str,
        signature: tuple[int, ...],
        segments: set[tuple[str, str]],
    ) -> None:
        digest = stable_hash(text)
        self.connection.execute(
            "INSERT OR IGNORE INTO documents(digest, source, signature) VALUES (?, ?, ?)",
            (digest, source, packed_signature(signature)),
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO bands(band, bucket, digest) VALUES (?, ?, ?)",
            [(band, bucket, digest) for band, bucket in signature_bands(signature)],
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO segments(kind, digest) VALUES (?, ?)",
            sorted(segments),
        )
        self.pending += 1
        if self.pending >= 250:
            self.checkpoint()
