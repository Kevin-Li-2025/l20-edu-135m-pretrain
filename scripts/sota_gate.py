#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.config import load_config
from l20_pretrain.eval_results import compare_task, extract_primary_metrics


def win_rate(candidate_dir: str, baseline_dir: str) -> tuple[int, int, float]:
    candidate = extract_primary_metrics(candidate_dir)
    baseline = extract_primary_metrics(baseline_dir)
    wins = 0
    comparable = 0
    for task, candidate_metric in candidate.items():
        baseline_metric = baseline.get(task)
        if baseline_metric is None or baseline_metric.metric != candidate_metric.metric:
            continue
        comparable += 1
        if compare_task(candidate_metric, baseline_metric) > 0:
            wins += 1
    return wins, comparable, wins / comparable if comparable else 0.0


def read_contamination(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail unless the project satisfies its SOTA gate.")
    parser.add_argument("--manifest", default="configs/sota_eval.yaml")
    parser.add_argument("--contamination-report", default="eval_results/contamination_report.json")
    parser.add_argument("--out", default="eval_results/sota_gate.json")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    candidate_cfg = load_config(manifest["candidate"]["config"])
    controlled_cfg = load_config(manifest["controlled_baseline"]["config"])
    gate = manifest["gate"]

    failures: list[str] = []
    if candidate_cfg.tokenizer_name != controlled_cfg.tokenizer_name:
        failures.append("controlled configs use different tokenizers")
    if candidate_cfg.dataset.name != controlled_cfg.dataset.name:
        failures.append("controlled configs use different datasets")
    if candidate_cfg.model.block_size != controlled_cfg.model.block_size:
        failures.append("controlled configs use different context lengths")

    tolerance = float(gate["token_budget_tolerance"])
    ratio = candidate_cfg.planned_tokens / controlled_cfg.planned_tokens
    if abs(ratio - 1.0) > tolerance:
        failures.append(f"token budget ratio {ratio:.6f} exceeds tolerance {tolerance}")

    contamination = read_contamination(args.contamination_report)
    if contamination is None:
        failures.append("missing contamination report")
    elif contamination.get("status") != gate["require_contamination_status"]:
        failures.append(f"contamination status is {contamination.get('status')}")

    controlled_result = None
    public_results = []
    try:
        controlled_result = win_rate(
            manifest["candidate"]["eval_dir"],
            manifest["controlled_baseline"]["eval_dir"],
        )
        if controlled_result[2] < float(gate["require_controlled_win_rate"]):
            failures.append(
                "controlled win rate "
                f"{controlled_result[2]:.3f} below {gate['require_controlled_win_rate']}"
            )
    except Exception as exc:
        failures.append(f"controlled comparison unavailable: {exc}")

    for baseline in manifest["public_baselines"]:
        try:
            public_results.append(
                {
                    "name": baseline["name"],
                    "wins_comparable_rate": win_rate(
                        manifest["candidate"]["eval_dir"],
                        baseline["eval_dir"],
                    ),
                }
            )
        except Exception as exc:
            public_results.append({"name": baseline["name"], "error": str(exc)})

    comparable_public = [
        result["wins_comparable_rate"][2]
        for result in public_results
        if "wins_comparable_rate" in result and result["wins_comparable_rate"][1] > 0
    ]
    if comparable_public:
        avg_public_rate = sum(comparable_public) / len(comparable_public)
        if avg_public_rate < float(gate["require_public_win_rate"]):
            failures.append(
                f"public average win rate {avg_public_rate:.3f} below {gate['require_public_win_rate']}"
            )
    else:
        avg_public_rate = 0.0
        failures.append("no public baseline comparisons available")

    payload = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "controlled_result": controlled_result,
        "public_results": public_results,
        "public_average_win_rate": avg_public_rate,
        "contamination": contamination,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
