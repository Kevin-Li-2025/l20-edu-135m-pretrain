from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def resolve_hf_revision(
    repo_id: str,
    *,
    repo_type: str,
    revision: str | None = None,
) -> str | None:
    """Resolve a floating Hub ref to the immutable commit used for an artifact."""

    if Path(repo_id).expanduser().exists():
        return revision
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return str(info.sha)


def verify_shard_directory(root: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    root = Path(root)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"shard metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("metadata.json does not contain a non-empty artifacts manifest")

    for name in ("train.bin", "val.bin"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"artifacts manifest is missing {name}")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        actual_bytes = path.stat().st_size
        expected_bytes = int(record.get("bytes", -1))
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"artifact size mismatch for {name}: expected={expected_bytes}, actual={actual_bytes}"
            )
        expected_tokens = metadata.get("train_tokens" if name == "train.bin" else "val_tokens")
        if expected_tokens is not None and actual_bytes != int(expected_tokens) * 4:
            raise RuntimeError(
                f"token count mismatch for {name}: metadata={expected_tokens}, bytes={actual_bytes}"
            )
        if verify_hashes:
            expected_hash = str(record.get("sha256", ""))
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"artifact SHA256 mismatch for {name}: expected={expected_hash}, actual={actual_hash}"
                )
    return metadata
