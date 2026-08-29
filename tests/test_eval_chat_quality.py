from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_chat_quality.py"
SPEC = importlib.util.spec_from_file_location("eval_chat_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_quality_checks_cover_exact_format_and_json() -> None:
    assert MODULE.evaluate_check("BLUE", {"type": "exact", "value": "BLUE"})
    assert not MODULE.evaluate_check("BLUE.", {"type": "exact", "value": "BLUE"})
    assert MODULE.evaluate_check(
        "- one\n- two\n- three",
        {"type": "bullet_count", "count": 3},
    )
    assert MODULE.evaluate_check(
        '{"name":"apple","count":3}',
        {"type": "json_keys", "keys": ["name", "count"]},
    )


def test_trigram_ratio_detects_repetition() -> None:
    assert MODULE.trigram_unique_ratio([1, 2, 3, 4, 5]) == 1.0
    assert MODULE.trigram_unique_ratio([1, 2, 3, 1, 2, 3, 1, 2, 3]) < 0.5
