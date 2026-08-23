#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from l20_pretrain.data_guard import BenchmarkContaminationIndex


DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate, concise AI assistant."


def render(messages: list[dict[str, str]], system_prompt: str) -> tuple[str, str]:
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}, *messages]
    full_parts: list[str] = []
    supervised_parts: list[str] = []
    for message in messages:
        segment = (
            f"<|im_start|>{message['role']}\n"
            f"{message['content'].strip()}<|im_end|>\n"
        )
        full_parts.append(segment)
        if message["role"] == "assistant":
            supervised_parts.append(segment)
    return "".join(full_parts), "".join(supervised_parts)


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{percentile}": float(np.percentile(values, percentile))
        for percentile in (50, 75, 90, 95, 99)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a prepared SFT JSONL artifact.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--chat-prompts", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    lengths: list[int] = []
    supervised_lengths: list[int] = []
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch:
            return
        rendered = [render(row["messages"], args.system_prompt) for row in batch]
        lengths.extend(
            tokenizer(
                [item[0] for item in rendered],
                add_special_tokens=False,
                return_length=True,
            )["length"]
        )
        supervised_lengths.extend(
            tokenizer(
                [item[1] for item in rendered],
                add_special_tokens=False,
                return_length=True,
            )["length"]
        )
        batch.clear()

    with args.train_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                batch.append(json.loads(line))
            if len(batch) >= args.batch_size:
                flush()
    flush()

    prompt_records: list[dict[str, str]] = []
    with args.chat_prompts.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_records.append(
                {
                    "benchmark": "chat_quality_v1",
                    "text": " ".join(
                        message["content"]
                        for message in row["messages"]
                        if message["role"] == "user"
                    ),
                }
            )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as temp:
        for record in prompt_records:
            temp.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp.flush()
        contamination = BenchmarkContaminationIndex(temp.name)
        hits: list[dict[str, Any]] = []
        with args.train_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                match = contamination.match(
                    "\n".join(message["content"] for message in row["messages"])
                )
                if match is not None:
                    hits.append({"digest": row["digest"], "match": match})

    values = np.asarray(lengths)
    supervised_values = np.asarray(supervised_lengths)
    payload = {
        "examples": len(values),
        "length": {
            **percentiles(values),
            "mean": float(values.mean()),
            "gt_1024": int((values > 1024).sum()),
            "gt_2048": int((values > 2048).sum()),
            "tokens_total": int(values.sum()),
        },
        "supervised": {
            **percentiles(supervised_values),
            "tokens_total": int(supervised_values.sum()),
            "fraction": float(supervised_values.sum() / values.sum()),
        },
        "chat_quality_contamination_hits": len(hits),
        "chat_quality_contamination_examples": hits[:10],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
