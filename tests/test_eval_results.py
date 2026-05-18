from pathlib import Path

from l20_pretrain.eval_results import extract_primary_metrics, render_markdown_comparison


def test_extract_primary_metrics(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        """
{
  "results": {
    "hellaswag": {"acc_norm,none": 0.42, "acc_norm_stderr,none": 0.01},
    "piqa": {"acc,none": 0.61}
  }
}
""".strip(),
        encoding="utf-8",
    )

    metrics = extract_primary_metrics(tmp_path)
    assert metrics["hellaswag"].metric == "acc_norm,none"
    assert metrics["hellaswag"].value == 0.42
    assert metrics["piqa"].metric == "acc,none"


def test_render_comparison() -> None:
    candidate = {"task": extract_primary_metrics_dict("acc,none", 0.7)}
    baseline = {"wide": {"task": extract_primary_metrics_dict("acc,none", 0.6)}}
    markdown = render_markdown_comparison("candidate", candidate, baseline)
    assert "candidate" in markdown
    assert "wide" in markdown
    assert "1/1" in markdown


def extract_primary_metrics_dict(metric: str, value: float):
    from l20_pretrain.eval_results import TaskMetric

    return TaskMetric(task="task", metric=metric, value=value)
