#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from l20_pretrain.rlvr_rewards import extract_gsm8k_gold


SYSTEM_PROMPT = (
    "You are a concise math reasoning assistant. Solve the problem step by step, "
    "do not repeat the problem statement, and end with `Final answer: <number>`."
)


def render_answer(answer: str) -> str:
    gold = extract_gsm8k_gold(answer)
    rationale = answer.split("####", 1)[0].strip()
    parts: list[str] = []
    if rationale:
        parts.append(rationale)
    if gold is not None:
        parts.append(f"Final answer: {gold}")
    return "\n".join(parts).strip()


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": str(row["question"]).strip()},
            {"role": "assistant", "content": render_answer(str(row["answer"]))},
        ]
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K chain-of-thought SFT JSONL files.")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config", default="main")
    parser.add_argument("--train-output", default="data/sft/gsm8k_cot_train.jsonl")
    parser.add_argument("--eval-output", default="data/sft/gsm8k_cot_eval.jsonl")
    parser.add_argument("--summary-output", default="data/sft/gsm8k_cot_summary.json")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=512)
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
        "system_prompt": SYSTEM_PROMPT,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
