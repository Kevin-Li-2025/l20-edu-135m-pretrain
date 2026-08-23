#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NAMES = ("lr5e5", "lr1e4", "lr2e4", "lr3e4", "lr6e4")


def json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "event" in payload:
                events.append(payload)
    return events


def latest_ifeval_result(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("**/results_*.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No IFEval result under {root}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))["results"]["ifeval"]


def metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def paired_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quality_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_pass_rate": values["prompt_pass_rate"],
        f"{prefix}_check_pass_rate": values["check_pass_rate"],
        f"{prefix}_stop_rate": values["stop_rate"],
        f"{prefix}_degenerate_repetition_rate": values[
            "degenerate_repetition_rate"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the skill-curriculum SFT LR pilot.")
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidates: dict[str, dict[str, Any]] = {}
    for name in NAMES:
        events = json_events(args.log_root / f"{name}.log")
        evaluations = [event for event in events if event["event"] == "eval"]
        done = [event for event in events if event["event"] == "done"]
        if not evaluations or not done:
            raise ValueError(f"Incomplete skill SFT training log: {name}")

        root = args.eval_root / name
        ifeval = latest_ifeval_result(root / "ifeval")
        primary_result = paired_result(root / "paired-vs-base-primary.json")
        development_result = paired_result(root / "paired-vs-base-development.json")
        primary = primary_result["aggregate"]
        development = development_result["aggregate"]
        significant_task_regressions = sorted(
            {
            name
            for result in (primary_result, development_result)
            for name, task in result.get("tasks", {}).items()
            if task["paired_bootstrap_ci"][1] < 0
            }
        )
        candidate = {
            "checkpoint": done[-1]["checkpoint"],
            "sft_eval_loss": evaluations[-1]["loss"],
            "sft_eval_perplexity": evaluations[-1]["perplexity"],
            **quality_metrics("chat", metrics(root / "chat_quality.json")),
            **quality_metrics("zh", metrics(root / "zh_skill_quality.json")),
            "ifeval_prompt_strict": ifeval["prompt_level_strict_acc,none"],
            "ifeval_instruction_strict": ifeval["inst_level_strict_acc,none"],
            "ifeval_prompt_loose": ifeval["prompt_level_loose_acc,none"],
            "ifeval_instruction_loose": ifeval["inst_level_loose_acc,none"],
            "ifeval_samples": ifeval["sample_len"],
            "primary_base_mean": primary["baseline_equal_task_mean"],
            "primary_candidate_mean": primary["candidate_equal_task_mean"],
            "primary_delta": primary["delta"],
            "primary_ci": primary["paired_bootstrap_ci"],
            "development_base_mean": development["baseline_equal_task_mean"],
            "development_candidate_mean": development["candidate_equal_task_mean"],
            "development_delta": development["delta"],
            "development_ci": development["paired_bootstrap_ci"],
            "significant_task_regressions": significant_task_regressions,
        }
        candidate["no_significant_primary_regression"] = candidate["primary_ci"][1] >= 0
        candidate["no_significant_development_regression"] = (
            candidate["development_ci"][1] >= 0
        )
        candidate["no_significant_task_regression"] = not significant_task_regressions
        candidates[name] = candidate

    retention_eligible = [
        name
        for name, candidate in candidates.items()
        if candidate["no_significant_primary_regression"]
        and candidate["no_significant_development_regression"]
        and candidate["no_significant_task_regression"]
    ]
    ranking = sorted(
        retention_eligible,
        key=lambda name: (
            candidates[name]["zh_prompt_pass_rate"],
            candidates[name]["ifeval_prompt_strict"],
            candidates[name]["chat_prompt_pass_rate"],
            -candidates[name]["zh_degenerate_repetition_rate"],
            -candidates[name]["chat_degenerate_repetition_rate"],
            candidates[name]["primary_delta"],
        ),
        reverse=True,
    )
    payload = {
        "status": "promotion_requires_heldout_review",
        "retention_rule": (
            "The upper bound of both aggregate paired 95% CIs and every individual-task "
            "paired 95% CI must be >= 0; final promotion also requires qualitative review "
            "and a held-out math check."
        ),
        "ranking_order": [
            "zh_prompt_pass_rate",
            "ifeval_prompt_strict",
            "chat_prompt_pass_rate",
            "lower_zh_repetition",
            "lower_chat_repetition",
            "primary_delta",
        ],
        "retention_eligible": retention_eligible,
        "ranking": ranking,
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
