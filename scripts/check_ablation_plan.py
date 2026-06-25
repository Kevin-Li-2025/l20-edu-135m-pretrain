#!/usr/bin/env python3
"""Validate the structured ablation plan used by the project report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "project",
    "purpose",
    "default_eval_suite",
    "experiments",
}

REQUIRED_EXPERIMENT_FIELDS = {
    "id",
    "priority",
    "hypothesis",
    "base_checkpoint",
    "changed_variables",
    "token_budget",
    "estimated_wall_clock_hours_l20",
    "success_metric",
    "minimum_effect_size",
    "required_gates",
    "expected_artifacts",
    "stop_criteria",
}

ALLOWED_PRIORITIES = {"P0", "P1", "P2"}


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def require_non_empty_list(value: Any, field: str, experiment_id: str | None = None) -> list[Any]:
    if not isinstance(value, list) or not value:
        where = f"experiment {experiment_id}: " if experiment_id else ""
        raise ValueError(f"{where}{field} must be a non-empty list")
    return value


def validate_experiment(exp: dict[str, Any], seen_ids: set[str]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_EXPERIMENT_FIELDS - set(exp)
    exp_id = str(exp.get("id", "<missing>"))
    if missing:
        errors.append(f"experiment {exp_id}: missing fields {sorted(missing)}")
        return errors

    if exp_id in seen_ids:
        errors.append(f"duplicate experiment id: {exp_id}")
    seen_ids.add(exp_id)

    if not exp_id.replace("_", "").isalnum():
        errors.append(f"experiment {exp_id}: id must be slug-like")
    if exp["priority"] not in ALLOWED_PRIORITIES:
        errors.append(f"experiment {exp_id}: invalid priority {exp['priority']!r}")
    if not isinstance(exp["hypothesis"], str) or len(exp["hypothesis"].split()) < 8:
        errors.append(f"experiment {exp_id}: hypothesis is too short")

    for field in ["changed_variables", "required_gates", "expected_artifacts", "stop_criteria"]:
        try:
            require_non_empty_list(exp[field], field, exp_id)
        except ValueError as exc:
            errors.append(str(exc))

    if not isinstance(exp["token_budget"], int) or exp["token_budget"] < 0:
        errors.append(f"experiment {exp_id}: token_budget must be a non-negative integer")
    if not isinstance(exp["estimated_wall_clock_hours_l20"], (int, float)) or exp["estimated_wall_clock_hours_l20"] <= 0:
        errors.append(f"experiment {exp_id}: estimated_wall_clock_hours_l20 must be positive")
    if not isinstance(exp["minimum_effect_size"], (int, float)) or exp["minimum_effect_size"] < 0:
        errors.append(f"experiment {exp_id}: minimum_effect_size must be non-negative")

    artifact_paths = exp.get("expected_artifacts", [])
    for artifact in artifact_paths:
        if not isinstance(artifact, str):
            errors.append(f"experiment {exp_id}: artifact path is not a string")
            continue
        if artifact.startswith("/") or ".." in Path(artifact).parts:
            errors.append(f"experiment {exp_id}: unsafe artifact path {artifact!r}")
        if not (artifact.startswith("results/") or artifact.startswith("docs/")):
            errors.append(f"experiment {exp_id}: artifact must live under results/ or docs/: {artifact}")

    return errors


def validate_plan(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(data)
    if missing:
        errors.append(f"missing top-level fields {sorted(missing)}")
        return errors

    try:
        require_non_empty_list(data["default_eval_suite"], "default_eval_suite")
        experiments = require_non_empty_list(data["experiments"], "experiments")
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    if not all(isinstance(task, str) and task for task in data["default_eval_suite"]):
        errors.append("default_eval_suite must contain non-empty strings")

    seen_ids: set[str] = set()
    for exp in experiments:
        if not isinstance(exp, dict):
            errors.append("each experiment must be an object")
            continue
        errors.extend(validate_experiment(exp, seen_ids))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="docs/project_report/ablation_plan.json",
        help="path to ablation plan JSON",
    )
    args = parser.parse_args()

    path = Path(args.path)
    try:
        errors = validate_plan(path)
    except json.JSONDecodeError as exc:
        return fail(f"invalid JSON: {exc}")
    except OSError as exc:
        return fail(str(exc))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"ablation plan ok: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
