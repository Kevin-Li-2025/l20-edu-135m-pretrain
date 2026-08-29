from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_smollm_promotion.py"
SPEC = importlib.util.spec_from_file_location("check_smollm_promotion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def paired(baseline: float, candidate: float, low: float, high: float) -> dict:
    return {
        "method": {"confidence": 0.975},
        "aggregate": {
            "baseline_equal_task_mean": baseline,
            "candidate_equal_task_mean": candidate,
            "delta": candidate - baseline,
            "paired_bootstrap_ci": [low, high],
        }
    }


def test_gate_requires_positive_ci_for_every_baseline() -> None:
    aggregate = {
        "candidate": "candidate",
        "means": {"candidate": 0.51, "smollm": 0.47, "smollm2": 0.49},
    }
    result = MODULE.check_promotion(
        aggregate,
        [
            ("smollm", paired(0.47, 0.51, 0.02, 0.06)),
            ("smollm2", paired(0.49, 0.51, 0.001, 0.039)),
        ],
    )
    assert result["status"] == "pass"


def test_gate_fails_when_positive_mean_is_not_significant() -> None:
    aggregate = {
        "candidate": "candidate",
        "means": {"candidate": 0.491, "smollm2": 0.49},
    }
    result = MODULE.check_promotion(
        aggregate,
        [("smollm2", paired(0.49, 0.491, -0.004, 0.006))],
    )
    assert result["status"] == "fail"
    assert not result["checks"]["smollm2"]["significant_improvement"]


def test_gate_rejects_mismatched_aggregate_and_paired_results() -> None:
    aggregate = {
        "candidate": "candidate",
        "means": {"candidate": 0.51, "smollm2": 0.49},
    }
    with pytest.raises(ValueError, match="candidate mismatch"):
        MODULE.check_promotion(
            aggregate,
            [("smollm2", paired(0.49, 0.50, 0.001, 0.019))],
        )


def test_gate_rejects_confidence_below_familywise_threshold() -> None:
    aggregate = {
        "candidate": "candidate",
        "means": {"candidate": 0.51, "smollm2": 0.49},
    }
    result = paired(0.49, 0.51, 0.001, 0.039)
    result["method"]["confidence"] = 0.95
    with pytest.raises(ValueError, match="below required"):
        MODULE.check_promotion(aggregate, [("smollm2", result)])
