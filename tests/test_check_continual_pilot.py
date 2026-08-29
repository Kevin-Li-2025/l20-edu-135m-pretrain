from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_continual_pilot.py"
SPEC = importlib.util.spec_from_file_location("check_continual_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def paired(tasks: dict[str, float], *, delta: float, low: float) -> dict:
    return {
        "method": {"confidence": 0.9833},
        "aggregate": {"delta": delta, "paired_bootstrap_ci": [low, delta + 0.01]},
        "tasks": {task: {"delta": value} for task, value in tasks.items()},
    }


def test_pilot_passes_only_with_primary_and_development_non_regression() -> None:
    primary = paired(
        {"arc_challenge": 0.01, "arc_easy": 0.02, "hellaswag": 0.01},
        delta=0.012,
        low=0.003,
    )
    development = paired({"boolq": 0.001, "sciq": 0.002}, delta=0.0015, low=-0.005)
    assert MODULE.check_continual_pilot(primary, development)["status"] == "pass"


def test_pilot_fails_on_material_single_task_regression() -> None:
    primary = paired(
        {"arc_challenge": 0.01, "arc_easy": 0.02, "hellaswag": 0.01},
        delta=0.012,
        low=0.003,
    )
    development = paired({"boolq": -0.006, "sciq": 0.02}, delta=0.007, low=-0.001)
    result = MODULE.check_continual_pilot(primary, development)
    assert result["status"] == "fail"
    assert result["checks"]["development"]["task_regressions"] == {"boolq": -0.006}


def test_pilot_fails_when_repair_task_does_not_improve() -> None:
    primary = paired(
        {"arc_challenge": 0.01, "arc_easy": 0.0, "hellaswag": 0.01},
        delta=0.008,
        low=0.001,
    )
    development = paired({"boolq": 0.0}, delta=0.0, low=-0.01)
    assert MODULE.check_continual_pilot(primary, development)["status"] == "fail"

