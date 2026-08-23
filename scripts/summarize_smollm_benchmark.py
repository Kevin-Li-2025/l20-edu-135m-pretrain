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
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("NAME and PATH must both be non-empty")
    return name, Path(raw_path)


def load_scores(path: Path) -> dict[str, float]:
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    scores: dict[str, float] = {}
    for result_file in files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            continue
        for task, (_, metric) in TASK_METRICS.items():
            value = results.get(task, {}).get(metric)
            if not isinstance(value, int | float):
                continue
            numeric = float(value)
            previous = scores.get(task)
            if previous is not None and previous != numeric:
                raise ValueError(
                    f"Conflicting {task} scores under {path}: {previous} vs {numeric}"
                )
            scores[task] = numeric
    missing = sorted(set(TASK_METRICS) - set(scores))
    if missing:
        raise ValueError(f"Missing tasks under {path}: {', '.join(missing)}")
    return scores


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def build_payload(
    score_by_name: dict[str, dict[str, float]],
    candidate_name: str,
    baseline_names: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task, (display, metric) in TASK_METRICS.items():
        row: dict[str, Any] = {"task": task, "display": display, "metric": metric}
        for name, scores in score_by_name.items():
            row[name] = scores[task]
        for baseline_name in baseline_names:
            row[f"gap_vs_{baseline_name}"] = (
                row[candidate_name] - row[baseline_name]
            )
        rows.append(row)

    means = {
        name: sum(scores.values()) / len(TASK_METRICS)
        for name, scores in score_by_name.items()
    }
    return {
        "candidate": candidate_name,
        "baselines": baseline_names,
        "tasks": rows,
        "task_metrics": {
            task: metric for task, (_, metric) in TASK_METRICS.items()
        },
        "scores": score_by_name,
        "means": means,
        "mean_gaps": {
            baseline: means[candidate_name] - means[baseline]
            for baseline in baseline_names
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    score_by_name = payload["scores"]
    rows = payload["tasks"]
    candidate_name = payload["candidate"]
    baseline_names = payload["baselines"]
    lines = ["# SmolLM Target Benchmark", ""]
    lines.append("| Task | Metric | " + " | ".join(score_by_name) + " |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(score_by_name)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display"],
                    row["metric"],
                    *[fmt(row[name]) for name in score_by_name],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Means", ""])
    for name, value in payload["means"].items():
        lines.append(f"- `{name}`: {fmt(value)}")
    if baseline_names:
        lines.extend(["", "## Candidate Gaps", ""])
        for baseline_name in baseline_names:
            lines.append(
                f"- `{candidate_name}` vs `{baseline_name}` mean gap: "
                f"{fmt(payload['mean_gaps'][baseline_name])}"
            )
            ranked = sorted(
                (
                    (row["display"], row[f"gap_vs_{baseline_name}"])
                    for row in rows
                ),
                key=lambda item: item[1],
            )
            for task_name, task_gap in ranked:
                lines.append(f"  - {task_name}: {fmt(task_gap)}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    score_by_name = payload["scores"]
    baseline_names = payload["baselines"]
    fieldnames = [
        "Task",
        "Metric",
        *score_by_name,
        *[f"Gap vs {name}" for name in baseline_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["tasks"]:
            writer.writerow(
                {
                    "Task": row["display"],
                    "Metric": row["metric"],
                    **{name: fmt(row[name]) for name in score_by_name},
                    **{
                        f"Gap vs {name}": fmt(row[f"gap_vs_{name}"])
                        for name in baseline_names
                    },
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the six SmolLM target lm-eval tasks and show gaps."
    )
    parser.add_argument("--result", action="append", required=True, type=parse_named_path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--out", type=Path, help="Write JSON only to this path.")
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    if args.out is not None and args.out_json is not None:
        parser.error("--out and --out-json cannot be used together")

    score_by_name = {name: load_scores(path) for name, path in args.result}
    if args.candidate not in score_by_name:
        parser.error(f"candidate {args.candidate!r} was not provided in --result")
    missing_baselines = [name for name in args.baseline if name not in score_by_name]
    if missing_baselines:
        parser.error(
            "baseline results were not provided: " + ", ".join(missing_baselines)
        )

    payload = build_payload(score_by_name, args.candidate, args.baseline)
    markdown = render_markdown(payload)

    explicit_outputs = any(
        output is not None
        for output in (args.out, args.out_md, args.out_json, args.out_csv)
    )
    out_json = args.out or args.out_json
    out_md = args.out_md
    out_csv = args.out_csv
    if not explicit_outputs:
        out_json = Path("eval_results/smollm_benchmark.json")
        out_md = Path("eval_results/smollm_benchmark.md")
        out_csv = Path("eval_results/smollm_benchmark.csv")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    if out_csv is not None:
        write_csv(out_csv, payload)

    if args.out is not None and args.out_md is None and args.out_csv is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
