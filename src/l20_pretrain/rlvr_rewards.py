from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_PREFIX_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FINAL_LINE_RE = re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", re.IGNORECASE)


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.10g}"


def extract_gsm8k_gold(answer: str) -> str | None:
    match = _ANSWER_PREFIX_RE.search(answer)
    if match:
        return normalize_numeric_answer(match.group(1))
    numbers = _NUMBER_RE.findall(answer)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def extract_numeric_answer(completion: str) -> str | None:
    boxed = _BOXED_RE.findall(completion)
    if boxed:
        normalized = normalize_numeric_answer(boxed[-1])
        if normalized is not None:
            return normalized

    tail_markers = [
        "final answer is",
        "answer is",
        "therefore",
        "so the answer is",
    ]
    lowered = completion.lower()
    for marker in tail_markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            numbers = _NUMBER_RE.findall(completion[idx:])
            if numbers:
                return normalize_numeric_answer(numbers[-1])

    numbers = _NUMBER_RE.findall(completion)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def gsm8k_correctness_reward(completion: str, answer: str) -> float:
    predicted = extract_numeric_answer(completion)
    gold = extract_gsm8k_gold(answer)
    return 1.0 if predicted is not None and predicted == gold else 0.0


def gsm8k_format_reward(completion: str, answer: str | None = None) -> float:
    del answer
    text = completion.strip()
    if not text:
        return 0.0
    reward = 0.0
    if _FINAL_LINE_RE.search(text) or "\\boxed{" in text:
        reward += 0.2
    elif extract_numeric_answer(text) is not None:
        reward += 0.2
    if 8 <= len(text.split()) <= 220:
        reward += 0.1
    return min(reward, 0.3)


def repetition_penalty(completion: str) -> float:
    words = re.findall(r"[a-zA-Z0-9]+", completion.lower())
    if len(words) < 24:
        return 0.0

    penalty = 0.0
    for ngram_size, threshold, weight in [(4, 0.22, 0.15), (8, 0.14, 0.2)]:
        ngrams = [tuple(words[idx : idx + ngram_size]) for idx in range(len(words) - ngram_size + 1)]
        if not ngrams:
            continue
        duplicate_ratio = 1.0 - (len(set(ngrams)) / len(ngrams))
        if duplicate_ratio > threshold:
            penalty += min(weight, (duplicate_ratio - threshold) * 2.0)

    sentences = [part.strip().lower() for part in re.split(r"[.!?\n]+", completion) if len(part.strip().split()) >= 6]
    if sentences:
        repeated_sentence_ratio = 1.0 - (len(set(sentences)) / len(sentences))
        if repeated_sentence_ratio > 0.25:
            penalty += min(0.25, repeated_sentence_ratio)
    return min(penalty, 0.5)


def gsm8k_reward_func(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        reward = gsm8k_correctness_reward(text, gold) + gsm8k_format_reward(text, gold) - repetition_penalty(text)
        rewards.append(max(0.0, reward))
    return rewards
