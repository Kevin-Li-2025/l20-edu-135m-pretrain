from __future__ import annotations

import json
from pathlib import Path
import importlib.util


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_rlvr_gsm8k_results.py"
_SPEC = importlib.util.spec_from_file_location("summarize_rlvr_gsm8k_results", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
render_markdown = _MODULE.render_markdown
summarize_result = _MODULE.summarize_result
with_deltas = _MODULE.with_deltas


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_result_tracks_accuracy_and_repetition(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [
            {"correct": True, "prediction": "4", "completion": "Let r be red. The answer is 4."},
            {
                "correct": False,
                "prediction": "5",
                "completion": (
                    "The red chickens produce eggs every day. "
                    "The red chickens produce eggs every day. "
                    "The red chickens produce eggs every day. "
                    "The answer is 5."
                ),
            },
        ],
    )

    summary = summarize_result("eval", path)
    assert summary["n"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["prediction_rate"] == 1.0
    assert summary["avg_repetition_penalty"] > 0.0


def test_render_markdown_includes_deltas(tmp_path: Path) -> None:
    before_path = tmp_path / "before.jsonl"
    after_path = tmp_path / "after.jsonl"
    write_jsonl(before_path, [{"correct": False, "prediction": "1", "completion": "The answer is 1."}])
    write_jsonl(after_path, [{"correct": True, "prediction": "2", "completion": "The answer is 2."}])

    summaries = with_deltas(
        [
            summarize_result("before", before_path),
            summarize_result("after", after_path),
        ]
    )
    markdown = render_markdown(summaries)
    assert "GSM8K RLVR Diagnostics" in markdown
    assert "100.00%" in markdown
