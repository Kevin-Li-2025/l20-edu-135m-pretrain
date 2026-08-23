#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("NAME and PATH must both be non-empty")
    return name, Path(raw_path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def finite_number(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def check_promotion(
    aggregate: dict[str, Any],
    paired: list[tuple[str, dict[str, Any]]],
    *,
    min_confidence: float = 0.975,
) -> dict[str, Any]:
    if not 0.0 < min_confidence < 1.0:
        raise ValueError("min_confidence must be between zero and one")
    candidate = aggregate.get("candidate")
    means = aggregate.get("means")
    if not isinstance(candidate, str) or not isinstance(means, dict):
        raise ValueError("aggregate result is missing candidate or means")
    candidate_mean = finite_number(means.get(candidate), "candidate mean")

    checks: dict[str, Any] = {}
    for baseline, payload in paired:
        if baseline not in means:
            raise ValueError(f"aggregate result is missing baseline {baseline!r}")
        baseline_mean = finite_number(means[baseline], f"{baseline} mean")
        paired_aggregate = payload.get("aggregate")
        method = payload.get("method")
        if not isinstance(paired_aggregate, dict):
            raise ValueError(f"paired result for {baseline} is missing aggregate")
        if not isinstance(method, dict):
            raise ValueError(f"paired result for {baseline} is missing method")
        confidence = finite_number(
            method.get("confidence"), f"{baseline} confidence"
        )
        if confidence < min_confidence:
            raise ValueError(
                f"paired result for {baseline} has confidence {confidence}, "
                f"below required {min_confidence}"
            )
        paired_baseline = finite_number(
            paired_aggregate.get("baseline_equal_task_mean"),
            f"{baseline} paired baseline mean",
        )
        paired_candidate = finite_number(
            paired_aggregate.get("candidate_equal_task_mean"),
            f"{baseline} paired candidate mean",
        )
        paired_delta = finite_number(
            paired_aggregate.get("delta"), f"{baseline} paired delta"
        )
        interval = paired_aggregate.get("paired_bootstrap_ci")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"paired result for {baseline} has an invalid CI")
        ci_low = finite_number(interval[0], f"{baseline} CI lower bound")
        ci_high = finite_number(interval[1], f"{baseline} CI upper bound")
        if ci_low > ci_high:
            raise ValueError(f"paired result for {baseline} has a reversed CI")
        if not math.isclose(baseline_mean, paired_baseline, abs_tol=1e-12):
            raise ValueError(f"aggregate/paired baseline mismatch for {baseline}")
        if not math.isclose(candidate_mean, paired_candidate, abs_tol=1e-12):
            raise ValueError(f"aggregate/paired candidate mismatch for {baseline}")
        aggregate_delta = candidate_mean - baseline_mean
        if not math.isclose(aggregate_delta, paired_delta, abs_tol=1e-12):
            raise ValueError(f"aggregate/paired delta mismatch for {baseline}")

        mean_improved = aggregate_delta > 0.0
        significant_improvement = ci_low > 0.0
        checks[baseline] = {
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "delta": aggregate_delta,
            "paired_bootstrap_ci": [ci_low, ci_high],
            "confidence": confidence,
            "mean_improved": mean_improved,
            "significant_improvement": significant_improvement,
            "passed": mean_improved and significant_improvement,
        }

    if not checks:
        raise ValueError("at least one paired baseline result is required")
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "pass" if passed else "fail",
        "criterion": (
            "candidate equal-task mean exceeds every baseline and each paired "
            f"bootstrap confidence interval (confidence >= {min_confidence}) "
            "has lower bound > 0"
        ),
        "candidate": candidate,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the SmolLM promotion gate.")
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--paired", action="append", required=True, type=parse_named_path)
    parser.add_argument("--min-confidence", type=float, default=0.975)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = check_promotion(
        read_json(args.aggregate),
        [(name, read_json(path)) for name, path in args.paired],
        min_confidence=args.min_confidence,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
