#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/l20_stage4_hq_crossdedup_8k")
    parser.add_argument("--out", default="eval_results/stage4_data_gate.json")
    args = parser.parse_args()
    root = Path(args.data_dir)
    failures: list[str] = []
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        failures.append("missing metadata.json")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text())

    target = int(metadata.get("target_tokens") or 0)
    train = int(metadata.get("train_tokens") or 0)
    val = int(metadata.get("val_tokens") or 0)
    if target < 3_000_000_000 or train < target:
        failures.append(f"incomplete train tokens: {train}/{target}")
    if val < 4_194_304:
        failures.append(f"insufficient validation tokens: {val}")
    for name in ("train.bin", "val.bin", "cross_source_guard.sqlite"):
        if not (root / name).is_file():
            failures.append(f"missing {name}")
    if (root / ".build_in_progress").exists() or (root / ".build_failed").exists():
        failures.append("data directory contains an unfinished or failed build marker")
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    if train_path.is_file() and train_path.stat().st_size != train * 4:
        failures.append(
            f"train.bin size mismatch: {train_path.stat().st_size} bytes for {train} tokens"
        )
    if val_path.is_file() and val_path.stat().st_size != val * 4:
        failures.append(
            f"val.bin size mismatch: {val_path.stat().st_size} bytes for {val} tokens"
        )
    if int(metadata.get("block_size") or 0) != 8192:
        failures.append(f"unexpected block size: {metadata.get('block_size')}")
    guard = metadata.get("data_guard") or {}
    if not guard.get("enabled"):
        failures.append("cross-source data guard was not enabled")
    if int(guard.get("contamination_ngram") or 0) != 13:
        failures.append("contamination n-gram is not 13")
    if float(guard.get("contamination_lcs_threshold") or 0) != 0.60:
        failures.append("contamination LCS threshold is not 0.60")

    quotas = metadata.get("quotas") or {}
    source_tokens = metadata.get("source_tokens") or {}
    for source, quota in quotas.items():
        actual = int(source_tokens.get(source) or 0)
        if actual < int(quota) * 0.98:
            failures.append(f"{source} below 98% quota: {actual}/{quota}")
    for source in metadata.get("sources") or []:
        estimate = source.get("unique_tokens_estimate")
        if estimate is None:
            continue
        cap = int(float(estimate) * float(source.get("max_epochs", 5.0)))
        actual = int(source_tokens.get(source["name"]) or 0)
        if actual > cap:
            failures.append(f"{source['name']} exceeds epoch cap: {actual}/{cap}")

    counters = metadata.get("counters") or {}
    contamination = sum(
        int(value) for key, value in counters.items() if key.startswith("contamination_")
    )
    db_counts = {}
    db_path = root / "cross_source_guard.sqlite"
    if db_path.is_file():
        with sqlite3.connect(db_path) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                failures.append(f"SQLite integrity check failed: {integrity}")
            for table in ("documents", "bands", "segments"):
                db_counts[table] = connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
        if db_counts.get("documents", 0) == 0:
            failures.append("empty cross-source dedup index")

    payload = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "target_tokens": target,
        "train_tokens": train,
        "val_tokens": val,
        "source_tokens": source_tokens,
        "reject_counters": counters,
        "contamination_rejections": contamination,
        "dedup_index_counts": db_counts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
