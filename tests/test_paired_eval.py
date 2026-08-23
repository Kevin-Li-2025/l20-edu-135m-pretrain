from __future__ import annotations

import json
from pathlib import Path

import pytest

from l20_pretrain.paired_eval import compare_paired_tasks, mcnemar_exact_p_value


def write_samples(root: Path, task: str, values: list[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"samples_{task}_fixed.jsonl"
    rows = [
        {"doc_id": index, "doc_hash": f"hash-{index}", "acc": value}
        for index, value in enumerate(values)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compare_paired_tasks_reports_exact_delta(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_samples(baseline, "winogrande", [0, 0, 1, 1])
    write_samples(candidate, "winogrande", [1, 0, 1, 1])

    result = compare_paired_tasks(
        baseline,
        candidate,
        tasks=["winogrande"],
        bootstrap_samples=200,
        seed=7,
    )

    assert result["aggregate"]["delta"] == pytest.approx(0.25)
    assert result["aggregate"]["pooled_wins"] == 1
    assert result["aggregate"]["pooled_losses"] == 0
    assert result["tasks"]["winogrande"]["examples"] == 4


def test_mcnemar_exact_known_small_case() -> None:
    assert mcnemar_exact_p_value(1, 0) == pytest.approx(1.0)
    assert mcnemar_exact_p_value(5, 0) == pytest.approx(0.0625)
