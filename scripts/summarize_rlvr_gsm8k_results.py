#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from l20_pretrain.rlvr_rewards import repetition_penalty
from l20_pretrain.sft_data import iter_local_jsonl


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("result name cannot be empty")
    return name, Path(raw_path)


def summarize_result(name: str, path: Path) -> dict[str, Any]:
    rows = list(iter_local_jsonl(path))
    completions = [str(row.get("completion", "")) for row in rows]
    word_counts = [len(text.split()) for text in completions]
    penalties = [repetition_penalty(text) for text in completions]
    correct = sum(1 for row in rows if bool(row.get("correct")))
    predicted = sum(1 for row in rows if row.get("prediction") is not None)
    n = len(rows)
    return {
        "name": name,
        "path": str(path),
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "prediction_rate": predicted / n if n else 0.0,
        "avg_words": mean(word_counts) if word_counts else 0.0,
        "avg_repetition_penalty": mean(penalties) if penalties else 0.0,
        "high_repetition_rate": sum(1 for value in penalties if value >= 0.2) / n if n else 0.0,
    }


def with_deltas(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summaries:
        return []
    baseline = summaries[0]
    rows: list[dict[str, Any]] = []
    for item in summaries:
        enriched = dict(item)
        enriched["accuracy_delta_vs_first"] = item["accuracy"] - baseline["accuracy"]
        enriched["repetition_delta_vs_first"] = item["avg_repetition_penalty"] - baseline["avg_repetition_penalty"]
        rows.append(enriched)
    return rows


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# GSM8K RLVR Diagnostics",
        "",
        "| Run | N | Correct | Accuracy | Delta | Prediction rate | Avg words | Avg repetition penalty | High repetition |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {name} | {n} | {correct} | {accuracy} | {delta} | {prediction_rate} | {avg_words:.1f} | {avg_rep:.4f} | {high_rep} |".format(
                name=item["name"],
                n=item["n"],
                correct=item["correct"],
                accuracy=format_pct(item["accuracy"]),
                delta=format_pct(item["accuracy_delta_vs_first"]),
                prediction_rate=format_pct(item["prediction_rate"]),
                avg_words=item["avg_words"],
                avg_rep=item["avg_repetition_penalty"],
                high_rep=format_pct(item["high_repetition_rate"]),
            )
        )
    lines.append("")
    lines.append("High repetition means `repetition_penalty >= 0.2` under the active RLVR reward diagnostics.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GSM8K RLVR exact-eval JSONL outputs.")
    parser.add_argument(
        "--result",
        action="append",
        type=parse_named_path,
        required=True,
        help="Named result as name=path, or just path to use the file stem as name.",
    )
    parser.add_argument("--out-json", default="eval_results/rlvr/gsm8k_rlvr_diagnostics.json")
    parser.add_argument("--out-md", default="eval_results/rlvr/gsm8k_rlvr_diagnostics.md")
    args = parser.parse_args()

    summaries = with_deltas([summarize_result(name, path) for name, path in args.result])
    payload = {"runs": summaries}

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summaries), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
