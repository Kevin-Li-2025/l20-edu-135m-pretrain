from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys


def build_fixture(root: Path) -> None:
    train = 3_000_000_000
    val = 4_194_304
    root.mkdir()
    with (root / "train.bin").open("wb") as handle:
        handle.truncate(train * 4)
    with (root / "val.bin").open("wb") as handle:
        handle.truncate(val * 4)
    with sqlite3.connect(root / "cross_source_guard.sqlite") as connection:
        for table in ("documents", "bands", "segments"):
            connection.execute(f"CREATE TABLE {table}(value TEXT)")
            connection.execute(f"INSERT INTO {table}(value) VALUES (?)", ("x",))
    metadata = {
        "target_tokens": train,
        "train_tokens": train,
        "val_tokens": val,
        "block_size": 8192,
        "quotas": {"source": train},
        "source_tokens": {"source": train},
        "sources": [
            {
                "name": "source",
                "unique_tokens_estimate": 1_000_000_000,
                "max_epochs": 4,
            }
        ],
        "counters": {},
        "data_guard": {
            "enabled": True,
            "contamination_ngram": 13,
            "contamination_lcs_threshold": 0.60,
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))


def main() -> None:
    root = Path("/tmp/stage4_gate_fixture")
    if root.exists():
        import shutil

        shutil.rmtree(root)
    build_fixture(root)
    command = [
        sys.executable,
        "scripts/check_stage4_data_gate.py",
        "--data-dir",
        str(root),
        "--out",
        "/tmp/stage4_gate_result.json",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(Path("/tmp/stage4_gate_result.json").read_text())["status"] == "pass"
    (root / ".build_failed").touch()
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 2
    assert "failed build marker" in failed.stdout
    print("stage4_data_gate_fault_injection_ok")


if __name__ == "__main__":
    main()
