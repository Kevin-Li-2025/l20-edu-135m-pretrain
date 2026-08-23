from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_bilingual_teacher_jobs.py"
SPEC = importlib.util.spec_from_file_location("prepare_bilingual_teacher_jobs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def row(source: str, index: int, prompt: str | None = None) -> dict:
    return {
        "source": source,
        "digest": f"{index:064x}",
        "messages": [
            {"role": "user", "content": prompt or f"prompt {index}"},
            {"role": "assistant", "content": "answer"},
        ],
    }


def test_selection_is_quota_bounded_and_order_independent() -> None:
    rows = [row("a", index) for index in range(10)] + [
        row("b", index + 100) for index in range(10)
    ]
    quotas = {"a": 3, "b": 2}

    first, stats = PREPARE.select_jobs(rows, quotas, seed=7, max_prompt_chars=100)
    second, _ = PREPARE.select_jobs(reversed(rows), quotas, seed=7, max_prompt_chars=100)

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert stats["selected"] == {"a": 3, "b": 2}
    assert all(item["mode"] == "translate_and_answer_zh" for item in first)


def test_selection_rejects_long_prompt_before_quota() -> None:
    rows = [row("a", 0, "x" * 101), row("a", 1), row("a", 2)]

    selected, stats = PREPARE.select_jobs(
        rows, {"a": 2}, seed=7, max_prompt_chars=100
    )

    assert len(selected) == 2
    assert stats["rejected"] == {"a:too_long": 1}
