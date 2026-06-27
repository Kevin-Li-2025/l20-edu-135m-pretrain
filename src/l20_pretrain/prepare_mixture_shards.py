from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import gc
import gzip
import json
import os
import pickle
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from .env import set_default_hf_home

set_default_hf_home()

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import DatasetConfig
from .data import fineweb_edu_sample_files, tokenize_without_specials
from .data_guard import CrossSourceDataGuard
from .prepare_shards import passes_dataset_score, write_tokens
from .quality import code_quality_filter, normalize_code_text, normalize_text, quality_filter, stable_hash


@dataclass
class SourceSpec:
    name: str
    kind: str
    dataset: str = ""
    config_name: str | None = None
    split: str = "train"
    text_column: str = "text"
    family: str = "text"
    weight: float = 1.0
    min_chars: int = 500
    max_chars: int = 45_000
    min_score: float | None = None
    min_int_score: int | None = None
    require_license_type: str | None = None
    include_metadata_header: bool = False
    tokenized_path: str | None = None
    tokenized_split: str = "train"
    sample_seed: int = 0
    download_workers: int = 8
    download_buffer_size: int = 64
    max_files: int | None = None
    file_offset: int = 0
    data_path: str | None = None
    cache_files_locally: bool = False
    repo_files: list[str] = field(default_factory=list)
    document_separator: str | None = None
    unique_tokens_estimate: int | None = None
    max_epochs: float = 5.0
    quality: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(**{key: value for key, value in raw.items() if value is not None})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mixed math/code/synthetic-textbook pretraining shards."
    )
    parser.add_argument("--recipe", required=True, help="YAML recipe with source weights and filters.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    return parser.parse_args()


def load_recipe(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle) or {}
    if not recipe.get("sources"):
        raise ValueError("Mixture recipe must define at least one source")
    return recipe


def create_s3_client() -> Any:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "Stack-Edu sources require boto3. Install project dependencies or run `pip install boto3`."
        ) from exc

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_stack_edu_blob(s3_client: Any, blob_id: str) -> str | None:
    try:
        obj = s3_client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
    except Exception as exc:
        error_code = getattr(getattr(exc, "response", None), "get", lambda *_: {})("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404"}:
            return None
        raise

    with gzip.GzipFile(fileobj=obj["Body"]) as handle:
        return handle.read().decode("utf-8", errors="ignore")


def format_code_document(example: dict[str, Any], text: str, *, include_metadata_header: bool) -> str:
    if not include_metadata_header:
        return text
    language = str(example.get("language") or "").strip()
    path = str(example.get("path") or "").strip()
    repo_name = str(example.get("repo_name") or "").strip()
    header = "\n".join(
        line
        for line in (
            f"Repository: {repo_name}" if repo_name else "",
            f"Path: {path}" if path else "",
            f"Language: {language}" if language else "",
        )
        if line
    )
    fence = language.lower().replace("csharp", "c#").replace("cpp", "cpp")
    return f"{header}\n\n```{fence}\n{text.rstrip()}\n```" if header else text


def load_streaming_dataset(source: SourceSpec) -> Any:
    if source.repo_files:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        urls = [f"{endpoint}/datasets/{source.dataset}/resolve/main/{path}" for path in source.repo_files]
        if all(path.endswith((".json.gz", ".jsonl.gz")) for path in source.repo_files):
            return iter_remote_json_gz(urls)
        if source.cache_files_locally and all(path.endswith(".parquet") for path in source.repo_files):
            token = os.environ.get("HF_TOKEN")
            files = [
                download_parquet_to_local_cache(
                    url=url,
                    repo_id=source.dataset,
                    filename=filename,
                    token=token,
                )
                for url, filename in zip(urls, source.repo_files, strict=True)
            ]
        else:
            files = urls
        builder = "parquet" if all(path.endswith(".parquet") for path in source.repo_files) else "json"
        return load_dataset(builder, data_files=files, split="train", streaming=True)

    dataset_config = DatasetConfig(
        name=source.dataset,
        config_name=source.config_name,
        split=source.split,
        streaming=True,
        shuffle_buffer=0,
    )
    direct_files = fineweb_edu_sample_files(dataset_config)
    if direct_files is None:
        if source.cache_files_locally:
            remote_source = replace(source, cache_files_locally=False)
            remote_files = hf_dataset_parquet_files(remote_source)
            if remote_files:
                return iter_cached_parquet_files(remote_files, source)
        direct_files = hf_dataset_parquet_files(source)
    if direct_files:
        return load_dataset(
            "parquet",
            data_files=direct_files,
            split=source.split,
            streaming=True,
        )

    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config_name:
        kwargs["name"] = source.config_name
    return load_dataset(**kwargs)


def iter_cached_parquet_files(
    urls: list[str], source: SourceSpec
) -> Iterator[dict[str, Any]]:
    token = os.environ.get("HF_TOKEN")
    for url in urls:
        filename = url.split("/resolve/main/", 1)[-1]
        local_path = Path(
            download_parquet_to_local_cache(
                url=url,
                repo_id=source.dataset,
                filename=filename,
                token=token,
            )
        )
        try:
            dataset = load_dataset(
                "parquet",
                data_files=[str(local_path)],
                split="train",
                streaming=True,
            )
            yield from dataset
        finally:
            # Keep locally cached parquet shards across controlled restarts.
            # Re-downloading multi-GB shards after every memory recycle is far
            # more expensive than the temporary disk footprint on this runner.
            pass


def iter_remote_json_gz(urls: list[str]) -> Iterator[dict[str, Any]]:
    import requests

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    session = requests.Session()
    for url in urls:
        with session.get(
            url,
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=(20, 300),
        ) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with gzip.GzipFile(fileobj=response.raw) as handle:
                for raw_line in handle:
                    try:
                        payload = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(payload, dict):
                        yield payload


def hf_dataset_parquet_files(source: SourceSpec) -> list[str] | None:
    if source.dataset not in {
        "HuggingFaceTB/dclm-edu",
        "HuggingFaceTB/smollm-corpus",
        "HuggingFaceTB/finemath",
    }:
        return None

    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(endpoint=endpoint, token=token)
    prefix = (source.data_path or source.config_name or "").strip("/")
    files: list[str] = []
    start = max(0, source.file_offset)
    stop = None if source.max_files is None else start + max(1, source.max_files)
    for item in api.list_repo_tree(
        source.dataset,
        repo_type="dataset",
        path_in_repo=prefix or None,
        recursive=True,
        expand=False,
    ):
        if not item.path.endswith(".parquet"):
            continue
        files.append(item.path)
        if stop is not None and len(files) >= stop:
            break
    if not files:
        raise RuntimeError(f"No parquet files found for {source.dataset}")
    files = sorted(files)[start:stop]
    if not files:
        raise RuntimeError(
            f"No parquet files left for {source.dataset} after file_offset={source.file_offset}"
        )
    if source.cache_files_locally:
        return [
            download_parquet_to_local_cache(
                url=f"{endpoint}/datasets/{source.dataset}/resolve/main/{path}",
                repo_id=source.dataset,
                filename=path,
                token=token,
            )
            for path in files
        ]
    return [f"{endpoint}/datasets/{source.dataset}/resolve/main/{path}" for path in files]


def download_parquet_to_local_cache(
    *,
    url: str,
    repo_id: str,
    filename: str,
    token: str | None,
) -> str:
    import requests

    cache_root = Path(os.environ.get("PARQUET_CACHE_DIR", "data/hf_parquet_cache"))
    relative = Path(re.sub(r"[^A-Za-z0-9_.-]+", "--", repo_id)) / filename
    target = cache_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # The initial Hub response includes x-linked-size. Following the redirect
    # makes a HEAD request to CAS, which is less reliable than ranged GETs.
    head = requests.head(url, headers=headers, allow_redirects=False, timeout=(10, 30))
    head.raise_for_status()
    expected_size = int(
        head.headers.get("x-linked-size")
        or head.headers.get("Content-Length")
        or 0
    )
    if target.exists() and (expected_size <= 0 or target.stat().st_size == expected_size):
        with target.open("rb") as handle:
            handle.seek(-4, os.SEEK_END)
            if handle.read(4) == b"PAR1":
                return str(target)
        target.unlink()

    part = target.with_suffix(target.suffix + ".part")
    if part.exists() and part.stat().st_size == expected_size > 0:
        part.replace(target)
        return str(target)
    if part.exists():
        part.unlink()

    print(
        json.dumps(
            {
                "event": "parquet_download_start",
                "repo": repo_id,
                "file": filename,
                "expected_size": expected_size,
                "target": str(target),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    config_path: str | None = None
    try:
        if token:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
                handle.write(f'header = "Authorization: Bearer {token}"\n')
                config_path = handle.name
            os.chmod(config_path, 0o600)
        if expected_size > 0:
            chunk_size = max(
                8 * 1024 * 1024,
                int(os.environ.get("PARQUET_RANGE_CHUNK_BYTES", str(64 * 1024 * 1024))),
            )
            worker_count = max(1, int(os.environ.get("PARQUET_RANGE_WORKERS", "4")))
            parts_dir = target.with_suffix(target.suffix + ".parts")
            parts_dir.mkdir(parents=True, exist_ok=True)
            ranges = [
                (index, start, min(expected_size - 1, start + chunk_size - 1))
                for index, start in enumerate(range(0, expected_size, chunk_size))
            ]

            def download_range(item: tuple[int, int, int]) -> tuple[int, int]:
                index, start, end = item
                expected_chunk_size = end - start + 1
                chunk_path = parts_dir / f"{index:06d}.part"
                if chunk_path.exists() and chunk_path.stat().st_size == expected_chunk_size:
                    return index, 0
                chunk_path.unlink(missing_ok=True)
                temp_path = chunk_path.with_suffix(".tmp")
                temp_path.unlink(missing_ok=True)
                command = [
                    "curl",
                    "--fail",
                    "--location",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    os.environ.get("PARQUET_CHUNK_MAX_SECONDS", "900"),
                    "--retry",
                    "12",
                    "--retry-all-errors",
                    "--retry-max-time",
                    "3600",
                    "--retry-delay",
                    "2",
                    "--speed-limit",
                    os.environ.get("PARQUET_MIN_BYTES_PER_SEC", "65536"),
                    "--speed-time",
                    os.environ.get("PARQUET_LOW_SPEED_SECONDS", "30"),
                    "--silent",
                    "--show-error",
                    "--range",
                    f"{start}-{end}",
                    "--output",
                    str(temp_path),
                ]
                if config_path:
                    command.extend(["--config", config_path])
                command.append(url)
                subprocess.run(command, check=True)
                actual_chunk_size = temp_path.stat().st_size
                if actual_chunk_size != expected_chunk_size:
                    temp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Range {start}-{end} for {repo_id}/{filename} returned "
                        f"{actual_chunk_size} bytes, expected {expected_chunk_size}"
                    )
                temp_path.replace(chunk_path)
                return index, expected_chunk_size

            completed_bytes = sum(
                (parts_dir / f"{index:06d}.part").stat().st_size
                for index, start, end in ranges
                if (parts_dir / f"{index:06d}.part").exists()
                and (parts_dir / f"{index:06d}.part").stat().st_size == end - start + 1
            )
            with ThreadPoolExecutor(max_workers=min(worker_count, len(ranges))) as executor:
                futures = [executor.submit(download_range, item) for item in ranges]
                for future in as_completed(futures):
                    _, chunk_bytes = future.result()
                    completed_bytes += chunk_bytes
                    print(
                        json.dumps(
                            {
                                "event": "parquet_download_progress",
                                "repo": repo_id,
                                "file": filename,
                                "completed_bytes": min(completed_bytes, expected_size),
                                "expected_size": expected_size,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )

            with part.open("wb") as output:
                for index, start, end in ranges:
                    chunk_path = parts_dir / f"{index:06d}.part"
                    if chunk_path.stat().st_size != end - start + 1:
                        raise RuntimeError(f"Invalid cached range file: {chunk_path}")
                    with chunk_path.open("rb") as source:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if part.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Merged parquet download for {repo_id}/{filename} has "
                    f"{part.stat().st_size} bytes, expected {expected_size}"
                )
            shutil.rmtree(parts_dir)
        else:
            command = [
                "curl",
                "--fail",
                "--location",
                "--connect-timeout",
                "10",
                "--retry",
                "12",
                "--retry-all-errors",
                "--retry-max-time",
                "3600",
                "--retry-delay",
                "5",
                "--speed-limit",
                os.environ.get("PARQUET_MIN_BYTES_PER_SEC", "65536"),
                "--speed-time",
                os.environ.get("PARQUET_LOW_SPEED_SECONDS", "30"),
                "--silent",
                "--show-error",
                "--output",
                str(part),
            ]
            if config_path:
                command.extend(["--config", config_path])
            command.append(url)
            subprocess.run(command, check=True)
    finally:
        if config_path:
            Path(config_path).unlink(missing_ok=True)

    actual_size = part.stat().st_size
    if expected_size > 0 and actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete parquet download for {repo_id}/{filename}: "
            f"got {actual_size}, expected {expected_size}"
        )
    with part.open("rb") as handle:
        handle.seek(-4, os.SEEK_END)
        if handle.read(4) != b"PAR1":
            raise RuntimeError(f"Invalid parquet footer for {repo_id}/{filename}")
    part.replace(target)
    print(
        json.dumps(
            {
                "event": "parquet_download_done",
                "repo": repo_id,
                "file": filename,
                "size": actual_size,
                "target": str(target),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return str(target)


def iter_source_examples(
    source: SourceSpec,
    *,
    s3_client: Any | None = None,
    dataset: Any | None = None,
) -> Iterator[dict[str, Any]]:
    dataset = dataset if dataset is not None else load_streaming_dataset(source)

    if source.kind == "stack_edu":
        if s3_client is None:
            raise ValueError("stack_edu source requires an S3 client")
        yield from iter_stack_edu_examples(dataset, source, s3_client=s3_client)
        return

    for example in dataset:
        if not isinstance(example, dict):
            continue
        if not passes_dataset_score(
            example,
            min_score=source.min_score,
            min_int_score=source.min_int_score,
        ):
            yield {"_reject": "score_reject", "_source": source.name}
            continue
        if source.require_license_type and example.get("license_type") != source.require_license_type:
            yield {"_reject": "license_reject", "_source": source.name}
            continue

        if source.document_separator:
            value = example.get(source.text_column)
            if isinstance(value, str) and source.document_separator in value:
                for document in value.split(source.document_separator):
                    document = document.strip()
                    if document:
                        enriched = dict(example)
                        enriched[source.text_column] = document
                        yield enriched
                continue
        yield example


def iter_stack_edu_examples(dataset: Any, source: SourceSpec, *, s3_client: Any) -> Iterator[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> Iterator[dict[str, Any]]:
        nonlocal buffer
        if not buffer:
            return
        batch = buffer
        buffer = []
        with ThreadPoolExecutor(max_workers=max(1, source.download_workers)) as executor:
            futures = {
                executor.submit(download_stack_edu_blob, s3_client, str(example["blob_id"])): example
                for example in batch
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    text = future.result()
                except Exception as exc:
                    yield {
                        "_reject": "download_error",
                        "_source": source.name,
                        "_error": str(exc)[:500],
                    }
                    continue
                if not text:
                    yield {"_reject": "download_failed", "_source": source.name}
                    continue
                enriched = dict(example)
                enriched[source.text_column] = format_code_document(
                    enriched,
                    text,
                    include_metadata_header=source.include_metadata_header,
                )
                yield enriched

    for example in dataset:
        if not isinstance(example, dict):
            continue
        if not passes_dataset_score(
            example,
            min_score=source.min_score,
            min_int_score=source.min_int_score,
        ):
            yield {"_reject": "score_reject", "_source": source.name}
            continue
        if source.require_license_type and example.get("license_type") != source.require_license_type:
            yield {"_reject": "license_reject", "_source": source.name}
            continue
        blob_id = example.get("blob_id")
        if not isinstance(blob_id, str) or not blob_id:
            yield {"_reject": "missing_blob_id", "_source": source.name}
            continue
        buffer.append(example)
        if len(buffer) >= source.download_buffer_size:
            yield from flush_buffer()
    yield from flush_buffer()


def get_text(example: dict[str, Any], text_column: str) -> str | None:
    value = example.get(text_column)
    return value if isinstance(value, str) else None


def passes_quality(text: str, source: SourceSpec) -> tuple[bool, str]:
    if source.family == "code":
        decision = code_quality_filter(text, min_chars=source.min_chars, **source.quality)
    else:
        decision = quality_filter(text, min_chars=source.min_chars, **source.quality)
    return decision.keep, decision.reason


def compute_train_quotas(target_tokens: int, sources: list[SourceSpec]) -> dict[str, int]:
    total_weight = sum(max(0.0, source.weight) for source in sources)
    if total_weight <= 0:
        raise ValueError("At least one source must have positive weight")

    quotas: dict[str, int] = {}
    assigned = 0
    for source in sources[:-1]:
        quota = int(target_tokens * max(0.0, source.weight) / total_weight)
        quotas[source.name] = quota
        assigned += quota
    quotas[sources[-1].name] = target_tokens - assigned
    for source in sources:
        if source.unique_tokens_estimate is None:
            continue
        epoch_cap = int(source.unique_tokens_estimate * source.max_epochs)
        if quotas[source.name] > epoch_cap:
            raise ValueError(
                f"{source.name} quota {quotas[source.name]:,} exceeds "
                f"{source.max_epochs:g} epochs of its estimated "
                f"{source.unique_tokens_estimate:,} unique tokens (cap {epoch_cap:,})"
            )
    return quotas


def write_token_array(handle: Any, tokens: np.ndarray) -> int:
    if tokens.size == 0:
        return 0
    array = np.asarray(tokens, dtype=np.uint32)
    array.tofile(handle)
    return int(array.size)


class ControlledRestart(RuntimeError):
    pass


def current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def release_unused_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def checkpoint_build(
    *,
    state_path: Path,
    train_handle: Any,
    val_handle: Any,
    guard: CrossSourceDataGuard | None,
    train_tokens: int,
    val_tokens: int,
    counters: Counter[str],
    source_counters: dict[str, Counter[str]],
    source_tokens: dict[str, int],
    source_seen: dict[str, int],
    source_overrides: dict[str, dict[str, int]],
    started_at: float,
) -> None:
    for handle in (train_handle, val_handle):
        handle.flush()
        os.fsync(handle.fileno())
    if guard is not None:
        guard.checkpoint()
    atomic_pickle(
        state_path,
        {
            "version": 1,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "counters": dict(counters),
            "source_counters": {
                name: dict(counter) for name, counter in source_counters.items()
            },
            "source_tokens": source_tokens,
            "source_seen": source_seen,
            "source_overrides": source_overrides,
            "elapsed_sec": time.time() - started_at,
        },
    )


def copy_tokenized_replay_source(
    source: SourceSpec,
    *,
    train_handle: Any,
    quota: int,
    target_tokens: int,
    train_tokens: int,
    block_size: int,
    counters: Counter[str],
    source_counter: Counter[str],
    source_tokens: dict[str, int],
    tokenizer: Any | None = None,
    eos_token_id: int | None = None,
    guard: CrossSourceDataGuard | None = None,
) -> int:
    if not source.tokenized_path:
        raise ValueError(f"{source.name} tokenized_replay source requires tokenized_path")
    source_path = Path(source.tokenized_path) / f"{source.tokenized_split}.bin"
    if not source_path.exists():
        raise FileNotFoundError(f"Tokenized replay shard not found: {source_path}")

    data = np.memmap(source_path, dtype=np.uint32, mode="r")
    total_tokens = int(data.shape[0])
    if total_tokens <= 0:
        raise ValueError(f"Tokenized replay shard is empty: {source_path}")
    epoch_cap = int(total_tokens * source.max_epochs)
    if quota > epoch_cap:
        raise ValueError(
            f"{source.name} replay quota {quota:,} exceeds {source.max_epochs:g} "
            f"epochs of {total_tokens:,} available tokens"
        )

    rng = np.random.default_rng(source.sample_seed)
    full_blocks = max(1, total_tokens // block_size)
    written_total = 0
    completed_epochs = 0
    while (
        source_tokens[source.name] < quota
        and train_tokens + written_total < target_tokens
        and completed_epochs < source.max_epochs
    ):
        block_ids = np.arange(full_blocks, dtype=np.int64)
        rng.shuffle(block_ids)
        for block_id in block_ids:
            remaining_quota = quota - source_tokens[source.name]
            remaining_target = target_tokens - train_tokens - written_total
            take = min(block_size, remaining_quota, remaining_target)
            if take <= 0:
                break
            start = int(block_id) * block_size
            if start + take > total_tokens:
                start = max(0, total_tokens - take)
            block = np.asarray(data[start : start + take], dtype=np.uint32)
            if guard is not None:
                if tokenizer is None or eos_token_id is None:
                    raise ValueError("Guarded replay requires tokenizer and eos_token_id")
                guarded_tokens: list[int] = []
                document: list[int] = []
                for token in block.tolist():
                    if token == eos_token_id:
                        if document:
                            text = tokenizer.decode(document, skip_special_tokens=True)
                            decision, signature, segments = guard.evaluate(text)
                            if decision.keep:
                                ids = tokenize_without_specials(tokenizer, decision.text)
                                guarded_tokens.extend(ids)
                                guarded_tokens.append(eos_token_id)
                                guard.add(
                                    text=decision.text,
                                    source=source.name,
                                    signature=signature,
                                    segments=segments,
                                )
                            else:
                                counters[decision.reason] += 1
                                source_counter[decision.reason] += 1
                        document = []
                    else:
                        document.append(int(token))
                remaining = min(remaining_quota, remaining_target)
                block = np.asarray(guarded_tokens[:remaining], dtype=np.uint32)
            written = write_token_array(train_handle, block)
            if written == 0:
                continue
            written_total += written
            source_tokens[source.name] += written
            counters["kept"] += 1
            counters["tokenized_replay_chunks"] += 1
            source_counter["kept"] += 1
            source_counter["train_kept"] += 1
            source_counter["tokenized_replay_chunks"] += 1
            if source_tokens[source.name] >= quota or train_tokens + written_total >= target_tokens:
                break
        completed_epochs += 1
    if source_tokens[source.name] < quota:
        shortfall = quota - source_tokens[source.name]
        counters["epoch_cap_shortfall_tokens"] += shortfall
        source_counter["epoch_cap_shortfall_tokens"] += shortfall
    return written_total


def main() -> None:
    args = parse_args()
    recipe = load_recipe(args.recipe)
    tokenizer_name = args.tokenizer or recipe.get("tokenizer", "AliceYin/l20-edu-135m")
    output_dir = Path(args.output_dir or recipe["output_dir"])
    target_tokens = int(args.target_tokens or recipe.get("target_tokens", 300_000_000))
    val_tokens_target = int(args.val_tokens or recipe.get("val_tokens", 2_097_152))
    block_size = int(args.block_size or recipe.get("block_size", 8192))
    report_interval = int(args.report_interval or recipe.get("report_interval", 1000))
    sources = [SourceSpec.from_dict(raw) for raw in recipe["sources"]]
    quotas = compute_train_quotas(target_tokens, sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    metadata_path = output_dir / "metadata.json"
    state_path = output_dir / "resume_state.pkl"
    guard_config = recipe.get("data_guard") or {}
    guard: CrossSourceDataGuard | None = None
    if guard_config.get("enabled", False):
        raw_index_path = Path(
            guard_config.get("index_path") or output_dir / "cross_source_guard.sqlite"
        )
        index_path = raw_index_path if raw_index_path.is_absolute() else Path(raw_index_path)
        guard = CrossSourceDataGuard(
            index_path,
            similarity_threshold=float(guard_config.get("similarity_threshold", 0.82)),
            max_duplicate_segment_fraction=float(
                guard_config.get("max_duplicate_segment_fraction", 0.30)
            ),
            contamination_path=guard_config.get("contamination_path"),
            contamination_ngram=int(guard_config.get("contamination_ngram", 13)),
            contamination_lcs_threshold=float(
                guard_config.get("contamination_lcs_threshold", 0.60)
            ),
        )
    saved_state: dict[str, Any] = {}
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume requested but state is missing: {state_path}")
        with state_path.open("rb") as handle:
            saved_state = pickle.load(handle)
        if int(saved_state.get("version", 0)) != 1:
            raise ValueError(f"Unsupported resume state version in {state_path}")

    counters: Counter[str] = Counter(saved_state.get("counters", {}))
    saved_source_counters = saved_state.get("source_counters", {})
    source_counters: dict[str, Counter[str]] = {
        source.name: Counter(saved_source_counters.get(source.name, {}))
        for source in sources
    }
    saved_source_tokens = saved_state.get("source_tokens", {})
    source_tokens: dict[str, int] = {
        source.name: int(saved_source_tokens.get(source.name, 0)) for source in sources
    }
    saved_source_seen = saved_state.get("source_seen", {})
    source_seen: dict[str, int] = {
        source.name: int(saved_source_seen.get(source.name, 0)) for source in sources
    }
    source_overrides: dict[str, dict[str, int]] = saved_state.get(
        "source_overrides", {}
    )
    for source in sources:
        override = source_overrides.get(source.name, {})
        file_offset_delta = int(override.get("file_offset_delta", 0))
        if file_offset_delta:
            source.file_offset += file_offset_delta
            if source.max_files is not None:
                source.max_files = max(1, source.max_files - file_offset_delta)
        if int(override.get("cache_files_locally", 0)):
            source.cache_files_locally = True
    train_tokens = int(saved_state.get("train_tokens", 0))
    val_tokens = int(saved_state.get("val_tokens", 0))
    started_at = time.time()
    s3_client = create_s3_client() if any(source.kind == "stack_edu" for source in sources) else None

    train_mode = "r+b" if args.resume else "w+b"
    val_mode = "r+b" if args.resume else "w+b"
    controlled_restart = False
    try:
        with train_path.open(train_mode) as train_handle, val_path.open(val_mode) as val_handle:
            if args.resume:
                train_handle.truncate(train_tokens * np.dtype(np.uint32).itemsize)
                val_handle.truncate(val_tokens * np.dtype(np.uint32).itemsize)
                train_handle.seek(0, os.SEEK_END)
                val_handle.seek(0, os.SEEK_END)
            for source in sources:
                if source_tokens[source.name] >= quotas[source.name]:
                    continue
                if source.kind == "tokenized_replay":
                    written = copy_tokenized_replay_source(
                        source,
                        train_handle=train_handle,
                        quota=quotas[source.name],
                        target_tokens=target_tokens,
                        train_tokens=train_tokens,
                        block_size=block_size,
                        counters=counters,
                        source_counter=source_counters[source.name],
                        source_tokens=source_tokens,
                        tokenizer=tokenizer,
                        eos_token_id=int(eos_token_id),
                        guard=guard,
                    )
                    train_tokens += written
                    checkpoint_build(
                        state_path=state_path,
                        train_handle=train_handle,
                        val_handle=val_handle,
                        guard=guard,
                        train_tokens=train_tokens,
                        val_tokens=val_tokens,
                        counters=counters,
                        source_counters=source_counters,
                        source_tokens=source_tokens,
                        source_seen=source_seen,
                        source_overrides=source_overrides,
                        started_at=started_at,
                    )
                    if train_tokens >= target_tokens and val_tokens >= val_tokens_target:
                        break
                    continue

                dataset = load_streaming_dataset(source)
                skip_examples = source_seen[source.name]
                if (
                    skip_examples
                    and source.kind != "stack_edu"
                    and not source.document_separator
                    and hasattr(dataset, "skip")
                ):
                    dataset = dataset.skip(skip_examples)
                    examples = iter_source_examples(
                        source, s3_client=s3_client, dataset=dataset
                    )
                else:
                    examples = iter_source_examples(
                        source, s3_client=s3_client, dataset=dataset
                    )
                    for skipped in range(skip_examples):
                        try:
                            next(examples)
                        except StopIteration:
                            raise RuntimeError(
                                f"{source.name} ended while restoring "
                                f"{skip_examples:,} examples"
                            ) from None
                        if skipped and skipped % 100_000 == 0:
                            release_unused_memory()
                if skip_examples:
                    print(
                        json.dumps(
                            {
                                "event": "resume_source",
                                "source": source.name,
                                "skipped_examples": skip_examples,
                                "train_tokens": train_tokens,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )

                last_checkpoint_seen = source_seen[source.name]
                for example in examples:
                    if (
                        source_seen[source.name] > last_checkpoint_seen
                        and source_seen[source.name] % max(1, args.checkpoint_interval) == 0
                    ):
                        checkpoint_build(
                            state_path=state_path,
                            train_handle=train_handle,
                            val_handle=val_handle,
                            guard=guard,
                            train_tokens=train_tokens,
                            val_tokens=val_tokens,
                            counters=counters,
                            source_counters=source_counters,
                            source_tokens=source_tokens,
                            source_seen=source_seen,
                            source_overrides=source_overrides,
                            started_at=started_at,
                        )
                        last_checkpoint_seen = source_seen[source.name]
                        release_unused_memory()
                        rss_bytes = current_rss_bytes()
                        if (
                            args.max_rss_gb > 0
                            and rss_bytes >= args.max_rss_gb * (1024**3)
                        ):
                            print(
                                json.dumps(
                                    {
                                        "event": "controlled_restart",
                                        "rss_bytes": rss_bytes,
                                        "max_rss_gb": args.max_rss_gb,
                                        "source": source.name,
                                        "source_seen": source_seen[source.name],
                                        "train_tokens": train_tokens,
                                    },
                                    ensure_ascii=True,
                                ),
                                flush=True,
                            )
                            raise ControlledRestart

                    counters["seen"] += 1
                    source_counters[source.name]["seen"] += 1
                    source_seen[source.name] += 1
                    if "_reject" in example:
                        reason = str(example["_reject"])
                        counters[reason] += 1
                        source_counters[source.name][reason] += 1
                        continue

                    raw_text = get_text(example, source.text_column)
                    if not raw_text:
                        counters["empty"] += 1
                        source_counters[source.name]["empty"] += 1
                        continue

                    if source.family == "code":
                        text = normalize_code_text(raw_text, max_chars=source.max_chars)
                    else:
                        text = normalize_text(raw_text, max_chars=source.max_chars)
                    keep, reason = passes_quality(text, source)
                    if not keep:
                        key = f"quality_{reason}"
                        counters[key] += 1
                        source_counters[source.name][key] += 1
                        continue

                    signature: tuple[int, ...] = ()
                    segments: set[tuple[str, str]] = set()
                    if guard is not None:
                        decision, signature, segments = guard.evaluate(text)
                        if not decision.keep:
                            counters[decision.reason] += 1
                            source_counters[source.name][decision.reason] += 1
                            if decision.benchmark:
                                key = f"contamination_{decision.benchmark}"
                                counters[key] += 1
                                source_counters[source.name][key] += 1
                            continue
                        text = decision.text
                        keep, reason = passes_quality(text, source)
                        if not keep:
                            key = f"post_guard_quality_{reason}"
                            counters[key] += 1
                            source_counters[source.name][key] += 1
                            continue

                    digest = stable_hash(text)
                    ids = tokenize_without_specials(tokenizer, text)
                    if len(ids) < 64:
                        counters["too_few_tokens"] += 1
                        source_counters[source.name]["too_few_tokens"] += 1
                        continue
                    ids.append(int(eos_token_id))

                    if val_tokens < val_tokens_target and int(digest[:8], 16) % 97 == 0:
                        val_tokens += write_tokens(val_handle, ids)
                        source_counters[source.name]["val_kept"] += 1
                    elif (
                        source_tokens[source.name] < quotas[source.name]
                        and train_tokens < target_tokens
                    ):
                        written = write_tokens(train_handle, ids)
                        train_tokens += written
                        source_tokens[source.name] += written
                        source_counters[source.name]["train_kept"] += 1
                    elif train_tokens >= target_tokens and val_tokens < val_tokens_target:
                        counters["train_full_val_scan"] += 1
                        source_counters[source.name]["train_full_val_scan"] += 1
                        continue
                    else:
                        break

                    counters["kept"] += 1
                    source_counters[source.name]["kept"] += 1
                    if guard is not None:
                        guard.add(
                            text=text,
                            source=source.name,
                            signature=signature,
                            segments=segments,
                        )
                    if counters["seen"] % report_interval == 0:
                        elapsed = max(time.time() - started_at, 1e-9)
                        print(
                            json.dumps(
                                {
                                    "event": "prepare_mixture",
                                    "source": source.name,
                                    "seen_docs": counters["seen"],
                                    "kept_docs": counters["kept"],
                                    "train_tokens": train_tokens,
                                    "val_tokens": val_tokens,
                                    "source_train_tokens": source_tokens,
                                    "tokens_per_sec": (
                                        train_tokens + val_tokens
                                    ) / elapsed,
                                    "rejects": {
                                        key: value
                                        for key, value in counters.items()
                                        if key not in {"seen", "kept"}
                                    },
                                },
                                ensure_ascii=True,
                            ),
                            flush=True,
                        )

                checkpoint_build(
                    state_path=state_path,
                    train_handle=train_handle,
                    val_handle=val_handle,
                    guard=guard,
                    train_tokens=train_tokens,
                    val_tokens=val_tokens,
                    counters=counters,
                    source_counters=source_counters,
                    source_tokens=source_tokens,
                    source_seen=source_seen,
                    source_overrides=source_overrides,
                    started_at=started_at,
                )
                release_unused_memory()
                if train_tokens >= target_tokens and val_tokens >= val_tokens_target:
                    break
    except ControlledRestart:
        controlled_restart = True
    finally:
        if guard is not None:
            guard.close()

    if controlled_restart:
        raise SystemExit(75)

    metadata = {
        "dtype": "uint32",
        "tokenizer": tokenizer_name,
        "recipe": str(args.recipe),
        "name": recipe.get("name"),
        "block_size": block_size,
        "target_tokens": target_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_blocks": train_tokens // block_size,
        "val_blocks": val_tokens // block_size,
        "quotas": quotas,
        "source_tokens": source_tokens,
        "sources": [source.__dict__ for source in sources],
        "counters": dict(counters),
        "source_counters": {name: dict(counter) for name, counter in source_counters.items()},
        "elapsed_sec": time.time() - started_at,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "data_guard": guard_config,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    state_path.unlink(missing_ok=True)
    print(json.dumps({"event": "done", **metadata}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
