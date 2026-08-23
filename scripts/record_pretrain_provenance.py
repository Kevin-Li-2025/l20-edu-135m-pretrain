#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, raw_value = value.split("=", 1)
    if not name or not raw_value:
        raise argparse.ArgumentTypeError("NAME and VALUE must both be non-empty")
    return name, raw_value


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> Iterator[tuple[str, Path]]:
    if path.is_file():
        yield path.name, path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and not candidate.is_symlink():
            yield candidate.relative_to(path).as_posix(), candidate


def build_manifest(
    inputs: list[tuple[str, str]], records: list[tuple[str, str]]
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for label, raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        for relative_path, file_path in iter_files(path):
            artifacts.append(
                {
                    "group": label,
                    "relative_path": relative_path,
                    "bytes": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),
                }
            )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records": dict(records),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a content-addressed manifest for a pretraining run."
    )
    parser.add_argument("--input", action="append", required=True, type=parse_assignment)
    parser.add_argument("--record", action="append", default=[], type=parse_assignment)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = build_manifest(args.input, args.record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "event": "provenance_complete",
                "output": str(args.out),
                "artifacts": len(payload["artifacts"]),
                "bytes": sum(item["bytes"] for item in payload["artifacts"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
