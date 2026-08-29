#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize steady-state throughput logs.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-step", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def read_json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def summarize_telemetry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    samples: dict[int, list[tuple[float, float, float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            gpu = int(fields[1])
            utilization = float(fields[2])
            power = float(fields[5])
            clock = float(fields[6])
        except ValueError:
            continue
        if utilization < 90:
            continue
        samples.setdefault(gpu, []).append((utilization, power, clock))
    return {
        str(gpu): {
            "samples": len(rows),
            "utilization_gpu_mean": statistics.mean(row[0] for row in rows),
            "power_w_mean": statistics.mean(row[1] for row in rows),
            "sm_clock_mhz_mean": statistics.mean(row[2] for row in rows),
        }
        for gpu, rows in sorted(samples.items())
        if rows
    }


def summarize_log(path: Path, min_step: int) -> dict[str, Any]:
    events = read_json_events(path)
    start = next((event for event in events if event.get("event") == "start"), {})
    train = [
        event
        for event in events
        if event.get("event") == "train"
        and int(event.get("step", -1)) >= min_step
        and float(event.get("tokens_per_sec_window", 0.0)) > 0
    ]
    if not train:
        return {"status": "failed", "log": str(path), "samples": 0}
    throughputs = [float(event["tokens_per_sec_window"]) for event in train]
    mfus = [float(event["mfu_pct"]) for event in train if "mfu_pct" in event]
    flops_per_token = start.get("flops_per_token_estimate")
    median_throughput = statistics.median(throughputs)
    result = {
        "status": "complete",
        "log": str(path),
        "samples": len(throughputs),
        "tokens_per_sec_median": median_throughput,
        "tokens_per_sec_mean": statistics.mean(throughputs),
        "tokens_per_sec_min": min(throughputs),
        "tokens_per_sec_max": max(throughputs),
        "tokens_per_sec_cv": (
            statistics.pstdev(throughputs) / statistics.mean(throughputs)
            if len(throughputs) > 1
            else 0.0
        ),
        "mfu_pct_median": statistics.median(mfus) if mfus else None,
        "world_size": start.get("world_size"),
        "block_size": start.get("block_size"),
        "tokens_per_step": start.get("tokens_per_step"),
        "flops_per_token_estimate": flops_per_token,
        "achieved_tflops_total": (
            median_throughput * float(flops_per_token) / 1e12
            if flops_per_token is not None
            else None
        ),
        "ddp_bucket_cap_mb": start.get("ddp_bucket_cap_mb"),
        "ddp_static_graph": start.get("ddp_static_graph"),
        "ddp_gradient_compression": start.get("ddp_gradient_compression"),
    }
    telemetry = summarize_telemetry(path.parent / "telemetry.csv")
    if telemetry:
        result["gpu_telemetry_at_or_above_90pct_util"] = telemetry
    return result


def summarize_root(root: Path, min_step: int) -> dict[str, Any]:
    cases = {
        path.parent.name: summarize_log(path, min_step)
        for path in sorted(root.glob("*/train.log"))
    }
    completed = {
        name: result for name, result in cases.items() if result["status"] == "complete"
    }
    best = (
        max(completed, key=lambda name: completed[name]["tokens_per_sec_median"])
        if completed
        else None
    )
    baseline = completed.get("baseline-2k-5g")
    if baseline:
        baseline_rate = baseline["tokens_per_sec_median"]
        for result in completed.values():
            result["delta_vs_baseline_pct"] = (
                100.0 * (result["tokens_per_sec_median"] / baseline_rate - 1.0)
            )
    return {"status": "complete" if completed else "failed", "best_case": best, "cases": cases}


def main() -> None:
    args = parse_args()
    payload = summarize_root(args.root, args.min_step)
    output_path = args.out or args.root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
