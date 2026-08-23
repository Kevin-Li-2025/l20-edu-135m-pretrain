#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import xxhash


_TOKENIZER: Any | None = None
_CONTAMINATION: Any | None = None


@dataclass(frozen=True)
class PartResult:
    index: int
    input_path: str
    train_path: str
    val_path: str
    train_tokens: int
    val_tokens: int
    seen_docs: int
    kept_docs: int
    duplicate_docs: int
    contaminated_docs: int
    source_filtered_docs: int
    source_tokens: dict[str, int]
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize many local parquet shards in parallel into packed uint32 data."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--glob", default="data/*.parquet")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 4))
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--flush-tokens", type=int, default=1_000_000)
    parser.add_argument("--target-tokens", type=int, default=9_800_000_000)
    parser.add_argument("--val-tokens", type=int, default=8_388_608)
    parser.add_argument("--val-denominator", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--contamination-index", type=Path, default=None)
    parser.add_argument("--contamination-ngram", type=int, default=13)
    parser.add_argument("--contamination-lcs-threshold", type=float, default=0.60)
    parser.add_argument(
        "--source-keep-rate",
        action="append",
        default=[],
        metavar="SOURCE=RATE",
        help=(
            "Deterministically retain RATE (0..1) of SOURCE documents. When at "
            "least one rate is supplied, unlisted sources are dropped. Repeat "
            "for every source in the desired mixture."
        ),
    )
    return parser.parse_args()


def parse_source_keep_rates(values: list[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --source-keep-rate {value!r}; expected SOURCE=RATE")
        source, raw_rate = value.split("=", 1)
        source = source.strip()
        if not source:
            raise ValueError("source name in --source-keep-rate cannot be empty")
        rate = float(raw_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"keep rate for {source!r} must be between 0 and 1")
        if source in rates:
            raise ValueError(f"duplicate --source-keep-rate for {source!r}")
        rates[source] = rate
    return rates


def _mix_uint64(value: int) -> int:
    """SplitMix64 finalizer to decorrelate sampling from validation splitting."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def keep_source_document(source: str, digest: int, rates: dict[str, float]) -> bool:
    if not rates:
        return True
    rate = rates.get(source, 0.0)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return _mix_uint64(digest) < int(rate * (1 << 64))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def initialize_worker(
    tokenizer_name: str,
    contamination_path: str | None,
    contamination_ngram: int,
    contamination_lcs_threshold: float,
    rayon_threads: int,
) -> None:
    global _TOKENIZER, _CONTAMINATION
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, rayon_threads))
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if contamination_path:
        from l20_pretrain.data_guard import BenchmarkContaminationIndex

        _CONTAMINATION = BenchmarkContaminationIndex(
            contamination_path,
            ngram=contamination_ngram,
            lcs_threshold=contamination_lcs_threshold,
        )


def valid_cached_part(metadata_path: Path) -> PartResult | None:
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = PartResult(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    train_path = Path(result.train_path)
    val_path = Path(result.val_path)
    if not train_path.is_file() or train_path.stat().st_size != result.train_tokens * 4:
        return None
    if not val_path.is_file() or val_path.stat().st_size != result.val_tokens * 4:
        return None
    return result


def flush_buffer(handle: Any, buffer: list[int]) -> int:
    if not buffer:
        return 0
    array = np.asarray(buffer, dtype=np.uint32)
    array.tofile(handle)
    count = int(array.size)
    buffer.clear()
    return count


def process_parquet_file(task: dict[str, Any]) -> dict[str, Any]:
    if _TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized")
    import pyarrow.parquet as pq

    started_at = time.time()
    index = int(task["index"])
    input_path = Path(task["input_path"])
    parts_dir = Path(task["parts_dir"])
    stem = f"{index:05d}-{input_path.stem}"
    train_path = parts_dir / f"{stem}.train.bin"
    val_path = parts_dir / f"{stem}.val.bin"
    metadata_path = parts_dir / f"{stem}.json"
    cached = valid_cached_part(metadata_path)
    if cached is not None:
        return asdict(cached)

    train_temporary = train_path.with_suffix(train_path.suffix + ".tmp")
    val_temporary = val_path.with_suffix(val_path.suffix + ".tmp")
    train_temporary.unlink(missing_ok=True)
    val_temporary.unlink(missing_ok=True)

    parquet = pq.ParquetFile(input_path)
    schema_names = set(parquet.schema_arrow.names)
    text_column = str(task["text_column"])
    source_column = str(task["source_column"])
    if text_column not in schema_names:
        raise ValueError(f"Missing {text_column!r} in {input_path}")
    columns = [text_column]
    if source_column in schema_names:
        columns.append(source_column)

    eos_token_id = _TOKENIZER.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")
    val_denominator = int(task["val_denominator"])
    batch_size = int(task["batch_size"])
    flush_tokens = int(task["flush_tokens"])
    source_keep_rates = {
        str(source): float(rate)
        for source, rate in dict(task.get("source_keep_rates", {})).items()
    }
    seen_hashes: set[int] = set()
    source_tokens: Counter[str] = Counter()
    train_buffer: list[int] = []
    val_buffer: list[int] = []
    train_tokens = 0
    val_tokens = 0
    seen_docs = 0
    kept_docs = 0
    duplicate_docs = 0
    contaminated_docs = 0
    source_filtered_docs = 0

    with train_temporary.open("wb") as train_handle, val_temporary.open("wb") as val_handle:
        for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            texts = record_batch.column(0).to_pylist()
            sources = (
                record_batch.column(1).to_pylist()
                if len(columns) == 2
                else ["unknown"] * len(texts)
            )
            accepted_texts: list[str] = []
            accepted_sources: list[str] = []
            accepted_hashes: list[int] = []
            for raw_text, raw_source in zip(texts, sources, strict=True):
                seen_docs += 1
                if not isinstance(raw_text, str) or not raw_text.strip():
                    continue
                digest = xxhash.xxh64_intdigest(raw_text.encode("utf-8"))
                if digest in seen_hashes:
                    duplicate_docs += 1
                    continue
                seen_hashes.add(digest)
                if _CONTAMINATION is not None and _CONTAMINATION.match(raw_text):
                    contaminated_docs += 1
                    continue
                source = str(raw_source or "unknown")
                if not keep_source_document(source, digest, source_keep_rates):
                    source_filtered_docs += 1
                    continue
                accepted_texts.append(raw_text)
                accepted_sources.append(source)
                accepted_hashes.append(digest)

            if not accepted_texts:
                continue
            encoded = _TOKENIZER(
                accepted_texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
            for ids, source, digest in zip(
                encoded, accepted_sources, accepted_hashes, strict=True
            ):
                if not ids:
                    continue
                token_count = len(ids) + 1
                target = val_buffer if digest % val_denominator == 0 else train_buffer
                target.extend(ids)
                target.append(int(eos_token_id))
                source_tokens[source] += token_count
                kept_docs += 1
            if len(train_buffer) >= flush_tokens:
                train_tokens += flush_buffer(train_handle, train_buffer)
            if len(val_buffer) >= flush_tokens:
                val_tokens += flush_buffer(val_handle, val_buffer)
        train_tokens += flush_buffer(train_handle, train_buffer)
        val_tokens += flush_buffer(val_handle, val_buffer)
        for handle in (train_handle, val_handle):
            handle.flush()
            os.fsync(handle.fileno())

    train_temporary.replace(train_path)
    val_temporary.replace(val_path)
    result = PartResult(
        index=index,
        input_path=str(input_path),
        train_path=str(train_path),
        val_path=str(val_path),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        seen_docs=seen_docs,
        kept_docs=kept_docs,
        duplicate_docs=duplicate_docs,
        contaminated_docs=contaminated_docs,
        source_filtered_docs=source_filtered_docs,
        source_tokens=dict(source_tokens),
        elapsed_sec=time.time() - started_at,
    )
    atomic_json(metadata_path, asdict(result))
    return asdict(result)


def copy_tokens(source_path: Path, output: Any, limit: int) -> int:
    remaining_bytes = max(0, limit) * 4
    copied_bytes = 0
    with source_path.open("rb") as source:
        while copied_bytes < remaining_bytes:
            chunk = source.read(min(64 * 1024 * 1024, remaining_bytes - copied_bytes))
            if not chunk:
                break
            usable = len(chunk) - len(chunk) % 4
            if usable <= 0:
                break
            output.write(chunk[:usable])
            copied_bytes += usable
            if usable != len(chunk):
                break
    return copied_bytes // 4


def output_is_complete(output_dir: Path) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    return (
        metadata.get("status") == "complete"
        and train_path.is_file()
        and val_path.is_file()
        and train_path.stat().st_size == int(metadata["train_tokens"]) * 4
        and val_path.stat().st_size == int(metadata["val_tokens"]) * 4
    )


def main() -> None:
    args = parse_args()
    source_keep_rates = parse_source_keep_rates(args.source_keep_rate)
    if args.workers <= 0 or args.batch_size <= 0 or args.val_denominator <= 1:
        raise ValueError("workers/batch-size must be positive and val-denominator > 1")
    if args.target_tokens <= 0 or args.val_tokens <= 0:
        raise ValueError("target-tokens and val-tokens must be positive")
    if output_is_complete(args.output_dir):
        print(json.dumps({"event": "reuse_complete", "output_dir": str(args.output_dir)}))
        return

    input_files = sorted(args.input_dir.glob(args.glob))
    if args.max_files is not None:
        input_files = input_files[: args.max_files]
    if not input_files:
        raise FileNotFoundError(f"No files match {args.input_dir / args.glob}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    tasks = [
        {
            "index": index,
            "input_path": str(path),
            "parts_dir": str(parts_dir),
            "text_column": args.text_column,
            "source_column": args.source_column,
            "val_denominator": args.val_denominator,
            "batch_size": args.batch_size,
            "flush_tokens": args.flush_tokens,
            "source_keep_rates": source_keep_rates,
        }
        for index, path in enumerate(input_files)
    ]

    results: list[PartResult] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        initializer=initialize_worker,
        initargs=(
            args.tokenizer,
            str(args.contamination_index) if args.contamination_index else None,
            args.contamination_ngram,
            args.contamination_lcs_threshold,
            args.rayon_threads,
        ),
    ) as executor:
        futures = [executor.submit(process_parquet_file, task) for task in tasks]
        for future in as_completed(futures):
            result = PartResult(**future.result())
            results.append(result)
            elapsed = max(time.time() - started_at, 1e-9)
            prepared_tokens = sum(item.train_tokens + item.val_tokens for item in results)
            print(
                json.dumps(
                    {
                        "event": "part_done",
                        "part": result.index,
                        "completed_parts": len(results),
                        "total_parts": len(tasks),
                        "prepared_tokens": prepared_tokens,
                        "tokens_per_sec": prepared_tokens / elapsed,
                        "part_tokens_per_sec": (
                            result.train_tokens + result.val_tokens
                        )
                        / max(result.elapsed_sec, 1e-9),
                    }
                ),
                flush=True,
            )

    results.sort(key=lambda item: item.index)
    rng = random.Random(args.seed)
    rng.shuffle(results)
    train_temporary = args.output_dir / "train.bin.tmp"
    val_temporary = args.output_dir / "val.bin.tmp"
    train_temporary.unlink(missing_ok=True)
    val_temporary.unlink(missing_ok=True)
    merged_train_tokens = 0
    merged_val_tokens = 0
    with train_temporary.open("wb") as train_handle:
        for result in results:
            merged_train_tokens += copy_tokens(
                Path(result.train_path),
                train_handle,
                args.target_tokens - merged_train_tokens,
            )
            if merged_train_tokens >= args.target_tokens:
                break
        train_handle.flush()
        os.fsync(train_handle.fileno())
    with val_temporary.open("wb") as val_handle:
        for result in results:
            merged_val_tokens += copy_tokens(
                Path(result.val_path),
                val_handle,
                args.val_tokens - merged_val_tokens,
            )
            if merged_val_tokens >= args.val_tokens:
                break
        val_handle.flush()
        os.fsync(val_handle.fileno())

    if merged_train_tokens < args.target_tokens:
        raise RuntimeError(
            f"Prepared only {merged_train_tokens:,} train tokens, below "
            f"target {args.target_tokens:,}"
        )
    if merged_val_tokens < args.val_tokens:
        raise RuntimeError(
            f"Prepared only {merged_val_tokens:,} val tokens, below "
            f"target {args.val_tokens:,}"
        )
    train_temporary.replace(args.output_dir / "train.bin")
    val_temporary.replace(args.output_dir / "val.bin")

    prepared_source_tokens: Counter[str] = Counter()
    for result in results:
        prepared_source_tokens.update(result.source_tokens)
    metadata = {
        "status": "complete",
        "dtype": "uint32",
        "tokenizer": args.tokenizer,
        "input_dir": str(args.input_dir),
        "input_glob": args.glob,
        "input_files": len(input_files),
        "block_size": args.block_size,
        "train_tokens": merged_train_tokens,
        "val_tokens": merged_val_tokens,
        "train_blocks": merged_train_tokens // args.block_size,
        "val_blocks": merged_val_tokens // args.block_size,
        "prepared_tokens": sum(item.train_tokens + item.val_tokens for item in results),
        "prepared_source_tokens": dict(prepared_source_tokens),
        "seen_docs": sum(item.seen_docs for item in results),
        "kept_docs": sum(item.kept_docs for item in results),
        "duplicate_docs": sum(item.duplicate_docs for item in results),
        "contaminated_docs": sum(item.contaminated_docs for item in results),
        "source_filtered_docs": sum(item.source_filtered_docs for item in results),
        "source_keep_rates": source_keep_rates,
        "contamination_index": (
            str(args.contamination_index) if args.contamination_index else None
        ),
        "workers": args.workers,
        "rayon_threads": args.rayon_threads,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "elapsed_sec": time.time() - started_at,
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps({"event": "done", **metadata}), flush=True)


if __name__ == "__main__":
    main()
