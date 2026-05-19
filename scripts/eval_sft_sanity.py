#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from l20_pretrain.env import set_default_hf_home

set_default_hf_home()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from l20_pretrain.train import get_device, get_dtype


SANITY_PROMPTS = [
    {
        "id": "capital_china",
        "instruction": "Answer with only the city name: What is the capital of China?",
        "expected_contains": ["beijing"],
    },
    {
        "id": "capital_uk",
        "instruction": "Answer with only the city name: What is the capital of the United Kingdom?",
        "expected_contains": ["london"],
    },
    {
        "id": "capital_uae",
        "instruction": "Answer with only the city name: What is the capital of the United Arab Emirates?",
        "expected_contains": ["abu dhabi"],
    },
    {
        "id": "capital_new_zealand",
        "instruction": "Answer with only the city name: What is the capital of New Zealand?",
        "expected_contains": ["wellington"],
    },
    {
        "id": "json_format",
        "instruction": 'Return valid JSON with keys "city" and "country" for Paris, France.',
        "json_required": True,
    },
    {
        "id": "short_story",
        "instruction": "Write a 5 sentence story about a student training a tiny language model.",
    },
    {
        "id": "concise_explanation",
        "instruction": "Explain in two short bullet points why supervised fine-tuning masks prompt tokens.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small SFT sanity eval.")
    parser.add_argument("checkpoint", help="HF repo id or local checkpoint directory.")
    parser.add_argument("--output", default="eval_results/sft_sanity/results.jsonl")
    parser.add_argument("--markdown-output", default="eval_results/sft_sanity/report.md")
    parser.add_argument("--system-prompt", default="You are a helpful, concise assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def format_prompt(system_prompt: str, instruction: str) -> str:
    return f"""### System:
{system_prompt}

### Instruction:
{instruction}

### Response:
"""


def response_only(prompt: str, text: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    marker = "### Response:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text.strip()


def score_response(item: dict[str, Any], response: str) -> dict[str, Any]:
    normalized = response.lower()
    score: dict[str, Any] = {"passed": None, "reason": "manual_review"}
    expected = item.get("expected_contains")
    if expected:
        passed = all(value in normalized for value in expected)
        return {"passed": passed, "reason": "contains_expected" if passed else "missing_expected"}
    if item.get("json_required"):
        try:
            json.loads(response)
        except json.JSONDecodeError:
            return {"passed": False, "reason": "invalid_json"}
        return {"passed": True, "reason": "valid_json"}
    return score


def load_model(checkpoint: str, dtype_name: str, device_name: str | None) -> tuple[Any, Any, torch.device]:
    device = get_device(device_name)
    dtype = get_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model, device


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for row in rows if row["score"]["passed"] is True)
    scored = sum(1 for row in rows if row["score"]["passed"] is not None)
    lines = [
        "# SFT Sanity Eval",
        "",
        f"Automatic checks passed: {passed} / {scored}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                "**Instruction**",
                "",
                row["instruction"],
                "",
                "**Response**",
                "",
                "```text",
                row["response"],
                "```",
                "",
                f"Score: `{row['score']['reason']}` / `{row['score']['passed']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tokenizer, model, device = load_model(args.checkpoint, args.dtype, args.device)

    rows: list[dict[str, Any]] = []
    for item in SANITY_PROMPTS:
        prompt = format_prompt(args.system_prompt, item["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response_only(prompt, decoded)
        row = {
            "id": item["id"],
            "instruction": item["instruction"],
            "response": response,
            "score": score_response(item, response),
        }
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(args.markdown_output, rows)
    print(f"Wrote {output_path}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
