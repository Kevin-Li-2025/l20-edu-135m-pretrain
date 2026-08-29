#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NAMES = ("lr5e5", "lr1e4", "lr3e4", "lr6e4", "lr1e3")


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
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    return payload["results"]["ifeval"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen five-arm SFT pilot.")
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
            raise ValueError(f"Incomplete SFT training log: {name}")
        chat = json.loads(
            (args.eval_root / name / "chat_quality.json").read_text(encoding="utf-8")
        )["metrics"]
        ifeval = latest_ifeval_result(args.eval_root / name / "ifeval")
        candidates[name] = {
            "checkpoint": done[-1]["checkpoint"],
            "sft_eval_loss": evaluations[-1]["loss"],
            "sft_eval_perplexity": evaluations[-1]["perplexity"],
            "chat_prompt_pass_rate": chat["prompt_pass_rate"],
            "chat_check_pass_rate": chat["check_pass_rate"],
            "chat_stop_rate": chat["stop_rate"],
            "chat_degenerate_repetition_rate": chat["degenerate_repetition_rate"],
            "ifeval_prompt_strict": ifeval["prompt_level_strict_acc,none"],
            "ifeval_instruction_strict": ifeval["inst_level_strict_acc,none"],
            "ifeval_prompt_loose": ifeval["prompt_level_loose_acc,none"],
            "ifeval_instruction_loose": ifeval["inst_level_loose_acc,none"],
            "ifeval_samples": ifeval["sample_len"],
        }

    eligible = {
        name: metrics
        for name, metrics in candidates.items()
        if metrics["chat_degenerate_repetition_rate"] <= 0.3125
    }
    ranking = sorted(
        eligible,
        key=lambda name: (
            eligible[name]["ifeval_prompt_strict"],
            eligible[name]["ifeval_instruction_strict"],
            eligible[name]["chat_prompt_pass_rate"],
            eligible[name]["chat_check_pass_rate"],
            -eligible[name]["sft_eval_loss"],
        ),
        reverse=True,
    )
    payload = {
        "status": "pilot_winner_pending_base_retention",
        "selection_order": [
            "ifeval_prompt_strict",
            "ifeval_instruction_strict",
            "chat_prompt_pass_rate",
            "chat_check_pass_rate",
            "sft_eval_loss",
        ],
        "winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
