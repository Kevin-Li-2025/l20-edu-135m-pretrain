import json
from pathlib import Path

import pytest

from scripts.verify_fineweb_1b_result import interpret_bf16_mfu, verify_result


ROOT = Path(__file__).resolve().parents[1]


def test_committed_fineweb_1b_receipt_is_internally_consistent() -> None:
    path = ROOT / "results/fineweb_1b/factorial_20260906.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    verify_result(payload, repo_root=ROOT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens_per_step", 1),
        ("planned_tokens", 1000000000),
        ("micro_batch_size", 0),
        ("max_steps", True),
    ],
)
def test_rejects_inconsistent_token_budget(field, value):
    payload = json.loads(
        (ROOT / "results/fineweb_1b/factorial_20260906.json").read_text()
    )
    payload["matched_controls"][field] = value
    with pytest.raises(ValueError):
        verify_result(payload, repo_root=ROOT)


def test_bf16_mfu_uses_dense_fp32_accumulation_peak():
    payload = json.loads(
        (ROOT / "results/fineweb_1b/factorial_20260906.json").read_text()
    )
    corrected = interpret_bf16_mfu(payload)
    assert corrected == pytest.approx(
        {
            "wide_cosine": 45.09574845997882,
            "wide_wsd": 45.11111207236178,
        }
    )
    # Preserve the original run record; this is interpretation, not rewritten telemetry.
    assert payload["cells"][2]["median_mfu_pct"] > 90


@pytest.mark.parametrize("peak", [0, -1, float("nan"), float("inf")])
def test_bf16_interpretation_rejects_invalid_denominator(peak):
    payload = {"matched_controls": {"mfu_peak_tflops_denominator": peak}, "cells": []}
    with pytest.raises(ValueError):
        interpret_bf16_mfu(payload)
