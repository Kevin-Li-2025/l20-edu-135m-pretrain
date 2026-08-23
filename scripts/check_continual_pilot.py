#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def finite(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def check_continual_pilot(
    primary: dict[str, Any],
    development: dict[str, Any],
    *,
    required_task_improvements: tuple[str, ...] = (
        "arc_challenge",
        "arc_easy",
        "hellaswag",
    ),
    max_task_regression: float = 0.005,
    min_confidence: float = 0.9833,
) -> dict[str, Any]:
    if max_task_regression < 0:
        raise ValueError("max_task_regression must be non-negative")
    checks: dict[str, Any] = {}
    for label, payload in (("primary", primary), ("development", development)):
        method = payload.get("method")
        aggregate = payload.get("aggregate")
        tasks = payload.get("tasks")
        if not isinstance(method, dict) or not isinstance(aggregate, dict) or not isinstance(tasks, dict):
            raise ValueError(f"{label} result is incomplete")
        confidence = finite(method.get("confidence"), f"{label} confidence")
        if confidence < min_confidence:
            raise ValueError(
                f"{label} confidence {confidence} is below required {min_confidence}"
            )
        delta = finite(aggregate.get("delta"), f"{label} aggregate delta")
        interval = aggregate.get("paired_bootstrap_ci")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{label} aggregate confidence interval is invalid")
        ci = [finite(interval[0], f"{label} CI lower"), finite(interval[1], f"{label} CI upper")]
        regressions = {
            task: finite(result.get("delta"), f"{label}/{task} delta")
            for task, result in tasks.items()
            if isinstance(result, dict)
            and finite(result.get("delta"), f"{label}/{task} delta") < -max_task_regression
        }
        checks[label] = {
            "delta": delta,
            "paired_bootstrap_ci": ci,
            "confidence": confidence,
            "task_regressions": regressions,
            "no_material_task_regression": not regressions,
        }

    primary_tasks = primary["tasks"]
    missing = [task for task in required_task_improvements if task not in primary_tasks]
    if missing:
        raise ValueError(f"primary result is missing required tasks: {missing}")
    required = {
        task: finite(primary_tasks[task].get("delta"), f"primary/{task} delta")
        for task in required_task_improvements
    }
    checks["required_task_improvements"] = required

    passed = (
        checks["primary"]["paired_bootstrap_ci"][0] > 0.0
        and checks["primary"]["no_material_task_regression"]
        and checks["development"]["delta"] >= 0.0
        and checks["development"]["no_material_task_regression"]
        and all(delta > 0.0 for delta in required.values())
    )
    return {
        "status": "pass" if passed else "fail",
        "criterion": (
            "primary equal-task mean improves significantly; ARC-Challenge, "
            "ARC-Easy, and HellaSwag each improve; development mean does not "
            "decline; and no individual task regresses beyond tolerance"
        ),
        "max_task_regression": max_task_regression,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a continual-pretraining pilot.")
    parser.add_argument("--primary-paired", required=True, type=Path)
    parser.add_argument("--development-paired", required=True, type=Path)
    parser.add_argument("--max-task-regression", type=float, default=0.005)
    parser.add_argument("--min-confidence", type=float, default=0.9833)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = check_continual_pilot(
        read_json(args.primary_paired),
        read_json(args.development_paired),
        max_task_regression=args.max_task_regression,
        min_confidence=args.min_confidence,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

