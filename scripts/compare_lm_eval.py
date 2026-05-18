#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.eval_results import extract_primary_metrics, render_markdown_comparison


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lm-eval result directories.")
    parser.add_argument("--candidate", required=True, type=parse_named_path)
    parser.add_argument("--baseline", action="append", default=[], type=parse_named_path)
    parser.add_argument("--out", default="eval_results/comparison.md")
    parser.add_argument("--json-out", default="eval_results/comparison.json")
    args = parser.parse_args()

    candidate_name, candidate_path = args.candidate
    candidate = extract_primary_metrics(candidate_path)
    baselines = {
        name: extract_primary_metrics(path)
        for name, path in args.baseline
    }

    markdown = render_markdown_comparison(candidate_name, candidate, baselines)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    json_payload = {
        "candidate": candidate_name,
        "tasks": {
            task: {"metric": metric.metric, "value": metric.value}
            for task, metric in candidate.items()
        },
        "baselines": {
            name: {
                task: {"metric": metric.metric, "value": metric.value}
                for task, metric in metrics.items()
            }
            for name, metrics in baselines.items()
        },
    }
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
