from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import summarize_smollm_benchmark as summary


def write_result(root: Path, score: float) -> None:
    root.mkdir(parents=True)
    results = {
        task: {metric: score}
        for task, (_, metric) in summary.TASK_METRICS.items()
    }
    (root / "results.json").write_text(
        json.dumps({"results": results}), encoding="utf-8"
    )


def test_json_only_output_is_compatible_with_promotion_gate(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    output = tmp_path / "comparison.json"
    write_result(candidate, 0.5)
    write_result(baseline, 0.4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_smollm_benchmark.py",
            "--result",
            f"candidate={candidate}",
            "--result",
            f"baseline={baseline}",
            "--candidate",
            "candidate",
            "--baseline",
            "baseline",
            "--out",
            str(output),
        ],
    )

    summary.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["means"]["candidate"] == pytest.approx(0.5)
    assert payload["means"]["baseline"] == pytest.approx(0.4)
    assert payload["mean_gaps"]["baseline"] == pytest.approx(0.1)
    assert not (tmp_path / "eval_results/smollm_benchmark.md").exists()


def test_legacy_output_flags_write_all_formats(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    write_result(candidate, 0.5)
    write_result(baseline, 0.4)
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_smollm_benchmark.py",
            "--result",
            f"candidate={candidate}",
            "--result",
            f"baseline={baseline}",
            "--candidate",
            "candidate",
            "--baseline",
            "baseline",
            "--out-json",
            str(json_path),
            "--out-md",
            str(markdown_path),
            "--out-csv",
            str(csv_path),
        ],
    )

    summary.main()

    assert json_path.is_file()
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# SmolLM Target Benchmark"
    )
    assert csv_path.read_text(encoding="utf-8").startswith("Task,Metric")
