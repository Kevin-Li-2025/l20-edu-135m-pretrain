import json

import numpy as np
import pytest

from l20_pretrain.provenance import artifact_record, verify_shard_directory


def _write_manifest(root) -> None:
    train = root / "train.bin"
    val = root / "val.bin"
    train.write_bytes(np.arange(12, dtype=np.uint32).tobytes())
    val.write_bytes(np.arange(4, dtype=np.uint32).tobytes())
    metadata = {
        "dtype": "uint32",
        "train_tokens": 12,
        "val_tokens": 4,
        "artifacts": {
            "train.bin": artifact_record(train),
            "val.bin": artifact_record(val),
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_verify_shard_directory_accepts_matching_manifest(tmp_path) -> None:
    _write_manifest(tmp_path)

    metadata = verify_shard_directory(tmp_path)

    assert metadata["train_tokens"] == 12


def test_verify_shard_directory_rejects_tampering(tmp_path) -> None:
    _write_manifest(tmp_path)
    train = tmp_path / "train.bin"
    original = train.read_bytes()
    train.write_bytes(b"x" * len(original))

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_shard_directory(tmp_path)
