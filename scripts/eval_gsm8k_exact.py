#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from l20_pretrain.rlvr_rewards import extract_gsm8k_gold, extract_numeric_answer
from l20_pretrain.sft_data import iter_local_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact GSM8K final-answer accuracy.")
    parser.add_argument("model")
    parser.add_argument("--data", default="data/rlvr/gsm8k_test.jsonl")
    parser.add_argument("--output", default="eval_results/rlvr/gsm8k_exact.jsonl")
    parser.add_argument("--summary-output", default="eval_results/rlvr/gsm8k_exact_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    rows = list(iter_local_jsonl(args.data))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows):
            prompt = str(row["prompt"])
            gold = extract_gsm8k_gold(str(row["answer"]))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            do_sample = args.temperature > 0
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            pred = extract_numeric_answer(completion)
            ok = pred is not None and pred == gold
            correct += int(ok)
            handle.write(
                json.dumps(
                    {
                        "question": row.get("question"),
                        "gold": gold,
                        "prediction": pred,
                        "correct": ok,
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {"model": args.model, "data": args.data, "n": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0}
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
