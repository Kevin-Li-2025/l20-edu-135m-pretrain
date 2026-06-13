#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TASK_METRICS = {
    "arc_challenge": ("ARC-Challenge", "acc_norm,none"),
    "arc_easy": ("ARC-Easy", "acc_norm,none"),
    "hellaswag": ("HellaSwag", "acc_norm,none"),
    "lambada_openai": ("LAMBADA OpenAI", "acc,none"),
    "piqa": ("PIQA", "acc_norm,none"),
    "winogrande": ("WinoGrande", "acc,none"),
}


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def find_result_json(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.rglob("results_*.json"))
    if not candidates:
        candidates = sorted(path.rglob("*.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            return candidate
    raise FileNotFoundError(f"No lm-eval result JSON found under {path}")


def load_scores(path: Path) -> dict[str, float | None]:
    result_json = find_result_json(path)
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    results = payload["results"]
    scores: dict[str, float | None] = {}
    for task, (_, metric) in TASK_METRICS.items():
        value = results.get(task, {}).get(metric)
        scores[task] = float(value) if isinstance(value, int | float) else None
    return scores


def mean_score(scores: dict[str, float | None]) -> float | None:
    values = [value for value in scores.values() if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the six SmolLM target lm-eval tasks and show gaps."
    )
    parser.add_argument("--result", action="append", default=[], type=parse_named_path)
    parser.add_argument("--candidate", default=None, help="Candidate model name for gap columns.")
    parser.add_argument("--baseline", action="append", default=[], help="Baseline names to compare against.")
    parser.add_argument("--out-md", default="eval_results/smollm_benchmark.md")
    parser.add_argument("--out-json", default="eval_results/smollm_benchmark.json")
    parser.add_argument("--out-csv", default="eval_results/smollm_benchmark.csv")
    args = parser.parse_args()

    if not args.result:
        raise SystemExit("At least one --result NAME=PATH is required.")

    score_by_name = {name: load_scores(path) for name, path in args.result}
    candidate_name = args.candidate or next(iter(score_by_name))
    if candidate_name not in score_by_name:
        raise SystemExit(f"Candidate {candidate_name!r} was not provided in --result.")
    baseline_names = [name for name in args.baseline if name in score_by_name]

    rows: list[dict[str, Any]] = []
    for task, (display, metric) in TASK_METRICS.items():
        row: dict[str, Any] = {
            "task": task,
            "display": display,
            "metric": metric,
        }
        for name, scores in score_by_name.items():
            row[name] = scores.get(task)
        for baseline_name in baseline_names:
            candidate_value = row.get(candidate_name)
            baseline_value = row.get(baseline_name)
            row[f"gap_vs_{baseline_name}"] = (
                None
                if candidate_value is None or baseline_value is None
                else float(candidate_value) - float(baseline_value)
            )
        rows.append(row)

    means = {name: mean_score(scores) for name, scores in score_by_name.items()}
    payload = {
        "candidate": candidate_name,
        "baselines": baseline_names,
        "tasks": rows,
        "means": means,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = ["Task", "Metric", *score_by_name.keys(), *[f"Gap vs {name}" for name in baseline_names]]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Task": row["display"],
                    "Metric": row["metric"],
                    **{name: fmt(row.get(name)) for name in score_by_name},
                    **{f"Gap vs {name}": fmt(row.get(f"gap_vs_{name}")) for name in baseline_names},
                }
            )

    lines = ["# SmolLM Target Benchmark", ""]
    lines.append("| Task | Metric | " + " | ".join(score_by_name.keys()) + " |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(score_by_name)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["display"], row["metric"], *[fmt(row.get(name)) for name in score_by_name]]
            )
            + " |"
        )
    lines.extend(["", "## Means", ""])
    for name, value in means.items():
        lines.append(f"- `{name}`: {fmt(value)}")
    if baseline_names:
        lines.extend(["", "## Candidate Gaps", ""])
        for baseline_name in baseline_names:
            baseline_mean = means.get(baseline_name)
            candidate_mean = means.get(candidate_name)
            gap = (
                None
                if baseline_mean is None or candidate_mean is None
                else candidate_mean - baseline_mean
            )
            lines.append(f"- `{candidate_name}` vs `{baseline_name}` mean gap: {fmt(gap)}")
            ranked = sorted(
                (
                    (row["display"], row.get(f"gap_vs_{baseline_name}"))
                    for row in rows
                    if row.get(f"gap_vs_{baseline_name}") is not None
                ),
                key=lambda item: item[1],
            )
            for task_name, task_gap in ranked:
                lines.append(f"  - {task_name}: {fmt(task_gap)}")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
