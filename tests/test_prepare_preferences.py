from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_posttrain_preferences.py"
SPEC = importlib.util.spec_from_file_location("prepare_posttrain_preferences", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def preference(prompt: str = "Question") -> dict:
    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Good answer"},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Bad answer"},
        ],
        "score_chosen": 8,
        "score_rejected": 3,
    }


def test_normalize_preference_splits_prompt_and_completions() -> None:
    row, reason = PREPARE.normalize_preference(
        preference(), max_prompt_chars=1000, max_completion_chars=1000
    )

    assert reason == "ok"
    assert row is not None
    assert row.prompt == [{"role": "user", "content": "Question"}]
    assert row.chosen == [{"role": "assistant", "content": "Good answer"}]
    assert row.rejected == [{"role": "assistant", "content": "Bad answer"}]
    assert row.score_chosen - row.score_rejected == 5


def test_normalize_preference_rejects_prompt_mismatch() -> None:
    raw = preference()
    raw["rejected"][0]["content"] = "Different question"

    row, reason = PREPARE.normalize_preference(
        raw, max_prompt_chars=1000, max_completion_chars=1000
    )

    assert row is None
    assert reason == "prompt_mismatch"


def test_preference_selection_is_order_independent() -> None:
    rows = []
    for index in range(20):
        row, _ = PREPARE.normalize_preference(
            preference(str(index)), max_prompt_chars=1000, max_completion_chars=1000
        )
        assert row is not None
        rows.append(row)

    forward = PREPARE.select_smallest(rows, 5, 42)
    backward = PREPARE.select_smallest(reversed(rows), 5, 42)

    assert [row.digest for row in forward] == [row.digest for row in backward]
