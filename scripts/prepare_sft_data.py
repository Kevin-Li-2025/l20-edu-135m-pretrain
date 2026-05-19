#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import json
from pathlib import Path
import random
import re
from typing import Any

from datasets import load_dataset

from l20_pretrain.sft_data import iter_local_jsonl


BAD_PHRASES = (
    "as an ai language model",
    "i am an ai language model",
    "i cannot browse",
    "i don't have access to",
    "i do not have access to",
)

CATEGORY_KEYWORDS = {
    "reasoning": (
        "why",
        "prove",
        "derive",
        "step by step",
        "calculate",
        "math",
        "logic",
        "explain",
        "compare",
    ),
    "factual_qa": (
        "what is",
        "who is",
        "where is",
        "when did",
        "capital",
        "define",
        "summarize",
    ),
    "writing": (
        "write",
        "story",
        "email",
        "poem",
        "letter",
        "rewrite",
        "draft",
        "tone",
    ),
    "format": (
        "json",
        "table",
        "bullet",
        "list",
        "yaml",
        "csv",
        "format",
        "schema",
    ),
    "coding": (
        "python",
        "javascript",
        "code",
        "function",
        "debug",
        "sql",
        "algorithm",
    ),
    "safety": (
        "safe",
        "harm",
        "dangerous",
        "illegal",
        "medical",
        "legal",
        "refuse",
    ),
}

MIXED_QUOTAS = {
    "reasoning": 0.22,
    "factual_qa": 0.20,
    "writing": 0.20,
    "format": 0.15,
    "coding": 0.10,
    "safety": 0.05,
    "general": 0.08,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare high-quality SFT JSONL splits.")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--eval-split", default="test_sft")
    parser.add_argument("--local-jsonl", default=None, help="Use a local JSONL source instead of Hugging Face.")
    parser.add_argument("--strategy", choices=("longest", "quality", "mixed"), required=True)
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--source-limit", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--max-response-chars", type=int, default=3000)
    parser.add_argument("--max-example-chars", type=int, default=12000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_messages(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None

    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = text(item.get("role")).lower()
        content = text(item.get("content"))
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages or None


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def dedupe_key(value: str) -> str:
    normalized = normalize_space(value.lower())
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return normalized[:1000]


def standardize_example(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = parse_messages(example.get("messages"))
    if messages:
        assistant_indices = [idx for idx, msg in enumerate(messages) if msg["role"] == "assistant"]
        if not assistant_indices:
            return None
        last_assistant = assistant_indices[-1]
        messages = messages[: last_assistant + 1]
        if not any(msg["role"] == "user" for msg in messages[:last_assistant]):
            return None
        return {"messages": messages}

    instruction = text(example.get("instruction")) or text(example.get("prompt"))
    extra_input = text(example.get("input"))
    response = text(example.get("output")) or text(example.get("response"))
    if not instruction or not response:
        return None
    user_content = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def prompt_and_response(example: dict[str, Any]) -> tuple[str, str] | None:
    messages = parse_messages(example.get("messages"))
    if not messages:
        return None
    assistant_indices = [idx for idx, msg in enumerate(messages) if msg["role"] == "assistant"]
    if not assistant_indices:
        return None
    target = assistant_indices[-1]
    prompt = "\n".join(msg["content"] for msg in messages[:target])
    response = messages[target]["content"]
    if not prompt or not response:
        return None
    return prompt, response


def repetition_penalty(response: str) -> float:
    words = re.findall(r"[a-zA-Z0-9']+", response.lower())
    if len(words) < 20:
        return 0.0
    unique_ratio = len(set(words)) / max(1, len(words))
    repeated_lines = len(response.splitlines()) - len({line.strip().lower() for line in response.splitlines() if line.strip()})
    penalty = max(0.0, 0.45 - unique_ratio) * 2.0
    return penalty + min(0.6, repeated_lines * 0.1)


def classify_category(prompt: str, response: str) -> str:
    combined = f"{prompt}\n{response[:500]}".lower()
    best_category = "general"
    best_count = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in combined)
        if count > best_count:
            best_category = category
            best_count = count
    return best_category


def quality_score(prompt: str, response: str) -> float:
    response_chars = len(response)
    response_words = len(re.findall(r"\S+", response))
    prompt_words = len(re.findall(r"\S+", prompt))

    if response_chars <= 0:
        return -100.0

    length_score = min(3.0, response_chars / 350.0)
    if response_chars > 1400:
        length_score -= min(1.5, (response_chars - 1400) / 1600.0)

    structure_score = 0.0
    if any(marker in response for marker in ("\n-", "\n1.", ":", "```")):
        structure_score += 0.35
    if response.count(".") + response.count("?") + response.count("!") >= 2:
        structure_score += 0.25

    prompt_score = min(1.0, prompt_words / 30.0)
    bad_penalty = sum(1.0 for phrase in BAD_PHRASES if phrase in response.lower())
    refusal_penalty = 0.25 if response.lower().startswith(("sorry", "i'm sorry", "i cannot")) else 0.0
    repeat_penalty = repetition_penalty(response)
    too_short_penalty = 2.0 if response_words < 20 else 0.0

    return length_score + structure_score + prompt_score - bad_penalty - refusal_penalty - repeat_penalty - too_short_penalty


def iter_source(args: argparse.Namespace, *, eval_split: bool = False) -> Iterable[dict[str, Any]]:
    if args.local_jsonl:
        return iter_local_jsonl(args.local_jsonl)
    kwargs: dict[str, Any] = {
        "path": args.dataset,
        "name": args.config,
        "split": args.eval_split if eval_split else args.split,
        "streaming": True,
    }
    return load_dataset(**kwargs)


def collect_candidates(args: argparse.Namespace, *, eval_split: bool = False) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_pairs: set[str] = set()
    for idx, raw in enumerate(iter_source(args, eval_split=eval_split)):
        if idx >= args.source_limit:
            break
        if not isinstance(raw, dict):
            continue
        standardized = standardize_example(raw)
        if standardized is None:
            continue
        rendered = prompt_and_response(standardized)
        if rendered is None:
            continue
        prompt, response = rendered
        if len(response) < args.min_response_chars or len(response) > args.max_response_chars:
            continue
        if len(json.dumps(standardized, ensure_ascii=False)) > args.max_example_chars:
            continue
        prompt_key = dedupe_key(prompt)
        pair_key = f"{prompt_key}::{dedupe_key(response)}"
        if prompt_key in seen_prompts or pair_key in seen_pairs:
            continue
        seen_prompts.add(prompt_key)
        seen_pairs.add(pair_key)
        score = quality_score(prompt, response)
        category = classify_category(prompt, response)
        standardized["metadata"] = {
            "source": args.dataset if not args.local_jsonl else args.local_jsonl,
            "source_split": args.eval_split if eval_split else args.split,
            "strategy": args.strategy,
            "score": round(score, 4),
            "category": category,
            "response_chars": len(response),
        }
        candidates.append(standardized)
    return candidates


def sorted_by_score(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("metadata", {}).get("score", 0.0)),
            int(item.get("metadata", {}).get("response_chars", 0)),
        ),
        reverse=True,
    )


def select_diverse(candidates: list[dict[str, Any]], target_size: int, *, category_cap: int | None = None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for item in sorted_by_score(candidates):
        category = str(item.get("metadata", {}).get("category", "general"))
        if category_cap is not None and counts[category] >= category_cap:
            continue
        selected.append(item)
        counts[category] += 1
        if len(selected) >= target_size:
            return selected
    for item in sorted_by_score(candidates):
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= target_size:
            break
    return selected


def select_mixed(candidates: list[dict[str, Any]], target_size: int) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted_by_score(candidates):
        by_category[str(item.get("metadata", {}).get("category", "general"))].append(item)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for category, ratio in MIXED_QUOTAS.items():
        quota = max(1, int(round(target_size * ratio)))
        for item in by_category.get(category, [])[:quota]:
            selected.append(item)
            selected_ids.add(id(item))
            if len(selected) >= target_size:
                return selected

    for item in sorted_by_score(candidates):
        if id(item) in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= target_size:
            break
    return selected


def select_candidates(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.strategy == "longest":
        return sorted(
            candidates,
            key=lambda item: (
                int(item.get("metadata", {}).get("response_chars", 0)),
                float(item.get("metadata", {}).get("score", 0.0)),
            ),
            reverse=True,
        )[: args.target_size]
    if args.strategy == "mixed":
        return select_mixed(candidates, args.target_size)
    category_cap = max(200, args.target_size // 3)
    return select_diverse(candidates, args.target_size, category_cap=category_cap)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    scores: list[float] = []
    response_chars: list[int] = []
    for row in rows:
        metadata = row.get("metadata", {})
        counts[str(metadata.get("category", "general"))] += 1
        scores.append(float(metadata.get("score", 0.0)))
        response_chars.append(int(metadata.get("response_chars", 0)))
    return {
        "size": len(rows),
        "categories": dict(sorted(counts.items())),
        "mean_score": round(sum(scores) / max(1, len(scores)), 4),
        "mean_response_chars": round(sum(response_chars) / max(1, len(response_chars)), 1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    candidates = collect_candidates(args)
    selected = select_candidates(candidates, args)
    if len(selected) < args.target_size:
        raise RuntimeError(f"Only selected {len(selected)} examples; requested {args.target_size}")

    eval_candidates = collect_candidates(args, eval_split=True)
    eval_selected = select_diverse(eval_candidates, args.eval_size, category_cap=max(80, args.eval_size // 3))
    if len(eval_selected) < args.eval_size:
        raise RuntimeError(f"Only selected {len(eval_selected)} eval examples; requested {args.eval_size}")

    write_jsonl(args.output, selected)
    write_jsonl(args.eval_output, eval_selected)

    summary = {
        "strategy": args.strategy,
        "source": args.local_jsonl or args.dataset,
        "split": args.split,
        "eval_split": args.eval_split,
        "source_limit": args.source_limit,
        "train": summarize(selected),
        "eval": summarize(eval_selected),
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
