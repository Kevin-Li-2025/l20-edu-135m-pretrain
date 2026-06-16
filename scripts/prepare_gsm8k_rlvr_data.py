#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


SYSTEM_PROMPT = (
    "Solve the grade-school math problem with concise reasoning. Do not repeat "
    "the problem statement. End the last line exactly as: Final answer: <number>."
)


def make_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nProblem:\n{question.strip()}\n\nSolution:"


def convert_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": make_prompt(str(row["question"])),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
    }


def write_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K JSONL files for RLVR/GRPO.")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config", default="main")
    parser.add_argument("--train-output", default="data/rlvr/gsm8k_train.jsonl")
    parser.add_argument("--eval-output", default="data/rlvr/gsm8k_test.jsonl")
    parser.add_argument("--summary-output", default="data/rlvr/gsm8k_summary.json")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    args = parser.parse_args()

    train = load_dataset(args.dataset, args.config, split="train")
    test = load_dataset(args.dataset, args.config, split="test")
    train_rows = [convert_row(row) for idx, row in enumerate(train) if args.max_train is None or idx < args.max_train]
    eval_rows = [convert_row(row) for idx, row in enumerate(test) if args.max_eval is None or idx < args.max_eval]

    write_jsonl(train_rows, Path(args.train_output))
    write_jsonl(eval_rows, Path(args.eval_output))
    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_output": args.train_output,
        "eval_output": args.eval_output,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
