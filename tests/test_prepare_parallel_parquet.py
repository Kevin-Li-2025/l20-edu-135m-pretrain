from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_parallel_parquet.py"
SPEC = importlib.util.spec_from_file_location("prepare_parallel_parquet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_source_keep_rates_rejects_invalid_values() -> None:
    assert MODULE.parse_source_keep_rates(["fineweb_edu=0.75"]) == {
        "fineweb_edu": 0.75
    }
    with pytest.raises(ValueError, match="expected SOURCE=RATE"):
        MODULE.parse_source_keep_rates(["fineweb_edu"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        MODULE.parse_source_keep_rates(["fineweb_edu=1.1"])
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.parse_source_keep_rates(["fineweb_edu=0.5", "fineweb_edu=0.6"])


def test_source_sampling_is_deterministic_and_drops_unlisted_sources() -> None:
    rates = {"fineweb_edu": 0.25, "dclm_edu": 1.0}
    first = [MODULE.keep_source_document("fineweb_edu", value, rates) for value in range(10_000)]
    second = [MODULE.keep_source_document("fineweb_edu", value, rates) for value in range(10_000)]

    assert first == second
    assert 2_300 < sum(first) < 2_700
    assert MODULE.keep_source_document("dclm_edu", 123, rates)
    assert not MODULE.keep_source_document("stack_edu", 123, rates)


def test_empty_source_rates_preserve_every_document() -> None:
    assert MODULE.keep_source_document("anything", 123, {})
