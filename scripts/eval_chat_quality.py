#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from l20_pretrain.sft_data import CHAT_TEMPLATE


def evaluate_check(text: str, check: dict[str, Any]) -> bool:
    kind = check["type"]
    if kind == "exact":
        return text.strip() == str(check["value"])
    if kind == "contains":
        return str(check["value"]).lower() in text.lower()
    if kind == "contains_any":
        return any(str(value).lower() in text.lower() for value in check["values"])
    if kind == "excludes":
        return str(check["value"]).lower() not in text.lower()
    if kind == "regex":
        return re.search(str(check["pattern"]), text, flags=re.IGNORECASE) is not None
    if kind == "bullet_count":
        bullets = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)
        return len(bullets) == int(check["count"])
    if kind == "json_keys":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(value, dict) and set(value) == set(check["keys"])
    raise ValueError(f"Unknown check type: {kind}")


def trigram_unique_ratio(token_ids: list[int]) -> float:
    if len(token_ids) < 3:
        return 1.0
    trigrams = [tuple(token_ids[index : index + 3]) for index in range(len(token_ids) - 2)]
    return len(set(trigrams)) / len(trigrams)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen deterministic chat quality suite.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in stop_ids:
        stop_ids.append(im_end_id)

    results: list[dict[str, Any]] = []
    for prompt in load_prompts(args.prompts):
        rendered = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(args.device)
        max_new_tokens = int(prompt.get("max_new_tokens", 128))
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )[0, inputs["input_ids"].shape[1] :].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        checks = [evaluate_check(text, check) for check in prompt.get("checks", [])]
        results.append(
            {
                "id": prompt["id"],
                "text": text,
                "checks": checks,
                "passed": bool(checks) and all(checks),
                "new_tokens": len(output_ids),
                "stopped": bool(output_ids) and output_ids[-1] in stop_ids,
                "trigram_unique_ratio": trigram_unique_ratio(output_ids),
            }
        )

    check_count = sum(len(row["checks"]) for row in results)
    passed_checks = sum(sum(row["checks"]) for row in results)
    metrics = {
        "model": args.model,
        "prompts": len(results),
        "prompt_pass_rate": sum(row["passed"] for row in results) / max(1, len(results)),
        "check_pass_rate": passed_checks / max(1, check_count),
        "stop_rate": sum(row["stopped"] for row in results) / max(1, len(results)),
        "degenerate_repetition_rate": sum(
            row["new_tokens"] >= 16 and row["trigram_unique_ratio"] < 0.5 for row in results
        )
        / max(1, len(results)),
    }
    payload = {"metrics": metrics, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
