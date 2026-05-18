from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREFERRED_METRICS = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "f1,none",
    "perplexity,none",
    "word_perplexity,none",
    "byte_perplexity,none",
)

LOWER_IS_BETTER = {
    "perplexity,none",
    "word_perplexity,none",
    "byte_perplexity,none",
    "loss,none",
}


@dataclass(frozen=True)
class TaskMetric:
    task: str
    metric: str
    value: float

    @property
    def higher_is_better(self) -> bool:
        return self.metric not in LOWER_IS_BETTER


def find_result_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*.json"))
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if isinstance(payload, dict) and "results" in payload:
            return candidate
    raise FileNotFoundError(f"No lm-eval result JSON found under {path}")


def load_lm_eval_payload(path: str | Path) -> dict[str, Any]:
    result_file = find_result_file(path)
    with result_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(f"{result_file} does not look like an lm-eval output")
    return payload


def choose_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    for metric in PREFERRED_METRICS:
        value = metrics.get(metric)
        if isinstance(value, int | float):
            return metric, float(value)
    for metric, value in metrics.items():
        if isinstance(value, int | float) and not metric.endswith("_stderr,none"):
            return metric, float(value)
    return None


def extract_primary_metrics(path: str | Path) -> dict[str, TaskMetric]:
    payload = load_lm_eval_payload(path)
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("lm-eval payload has non-dict results")

    output: dict[str, TaskMetric] = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        chosen = choose_metric(metrics)
        if chosen is None:
            continue
        metric, value = chosen
        output[str(task)] = TaskMetric(task=str(task), metric=metric, value=value)
    return output


def compare_task(candidate: TaskMetric, baseline: TaskMetric) -> int:
    if candidate.metric != baseline.metric:
        raise ValueError(
            f"Metric mismatch for {candidate.task}: {candidate.metric} vs {baseline.metric}"
        )
    if candidate.value == baseline.value:
        return 0
    if candidate.higher_is_better:
        return 1 if candidate.value > baseline.value else -1
    return 1 if candidate.value < baseline.value else -1


def format_score(metric: TaskMetric | None) -> str:
    if metric is None:
        return ""
    return f"{metric.value:.4f}"


def render_markdown_comparison(
    candidate_name: str,
    candidate: dict[str, TaskMetric],
    baselines: dict[str, dict[str, TaskMetric]],
) -> str:
    tasks = sorted(candidate)
    lines = [
        "# lm-eval Comparison",
        "",
        f"Candidate: `{candidate_name}`",
        "",
    ]
    header = ["Task", "Metric", candidate_name, *baselines.keys()]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for task in tasks:
        candidate_metric = candidate[task]
        row = [task, candidate_metric.metric, format_score(candidate_metric)]
        for baseline_metrics in baselines.values():
            row.append(format_score(baseline_metrics.get(task)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Win Rates", ""])
    for baseline_name, baseline_metrics in baselines.items():
        comparable = 0
        wins = 0
        for task, candidate_metric in candidate.items():
            baseline_metric = baseline_metrics.get(task)
            if baseline_metric is None or baseline_metric.metric != candidate_metric.metric:
                continue
            comparable += 1
            if compare_task(candidate_metric, baseline_metric) > 0:
                wins += 1
        rate = wins / comparable if comparable else 0.0
        lines.append(f"- `{baseline_name}`: {wins}/{comparable} = {rate:.3f}")
    lines.append("")
    return "\n".join(lines)
