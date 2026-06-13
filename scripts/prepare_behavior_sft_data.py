#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


CAPITALS = [
    ("Australia", "Canberra"),
    ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"),
    ("China", "Beijing"),
    ("Egypt", "Cairo"),
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("India", "New Delhi"),
    ("Ireland", "Dublin"),
    ("Italy", "Rome"),
    ("Japan", "Tokyo"),
    ("New Zealand", "Wellington"),
    ("Norway", "Oslo"),
    ("Portugal", "Lisbon"),
    ("South Korea", "Seoul"),
    ("Spain", "Madrid"),
    ("Sweden", "Stockholm"),
    ("Switzerland", "Bern"),
    ("United Arab Emirates", "Abu Dhabi"),
    ("United Kingdom", "London"),
]

JSON_ROWS = [
    ("Paris", "France"),
    ("Tokyo", "Japan"),
    ("Wellington", "New Zealand"),
    ("London", "United Kingdom"),
    ("Abu Dhabi", "United Arab Emirates"),
    ("Beijing", "China"),
    ("Ottawa", "Canada"),
    ("Canberra", "Australia"),
]

TWO_BULLET_ROWS = [
    (
        "why supervised fine-tuning masks prompt tokens",
        [
            "It trains the model to predict the assistant answer, not copy the user prompt.",
            "It keeps the loss focused on the behavior we want the assistant to learn.",
        ],
    ),
    (
        "why validation loss matters",
        [
            "It estimates how well the model generalizes beyond the training examples.",
            "It helps catch overfitting even when training loss keeps improving.",
        ],
    ),
    (
        "why short answers are useful in QA",
        [
            "They reduce irrelevant text around the requested fact.",
            "They make automatic checking and user reading easier.",
        ],
    ),
    (
        "why JSON outputs need exact syntax",
        [
            "Downstream tools often parse JSON strictly.",
            "A single extra sentence can make the response unusable.",
        ],
    ),
]

STORY_TOPICS = [
    "a student training a tiny language model",
    "a researcher debugging a GPU job",
    "a developer writing an eval harness",
    "a team releasing a small open model",
]


def add(rows: list[dict[str, str]], instruction: str, response: str) -> None:
    rows.append({"instruction": instruction, "input": "", "output": response})


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for country, city in CAPITALS:
        add(rows, f"Answer with only the city name: What is the capital of {country}?", city)
        add(rows, f"What is the capital of {country}? Reply with only the city.", city)
        add(rows, f"Give only the answer, no explanation: capital of {country}.", city)

    for city, country in JSON_ROWS:
        add(
            rows,
            f'Return valid JSON with keys "city" and "country" for {city}, {country}.',
            json.dumps({"city": city, "country": country}, ensure_ascii=False),
        )
        add(
            rows,
            f"Create one compact JSON object for the city {city} in {country}.",
            json.dumps({"city": city, "country": country}, ensure_ascii=False),
        )

    for topic, bullets in TWO_BULLET_ROWS:
        add(
            rows,
            f"Explain in two short bullet points {topic}.",
            "\n".join(f"- {item}" for item in bullets),
        )
        add(
            rows,
            f"Give exactly two concise bullets about {topic}.",
            "\n".join(f"- {item}" for item in bullets),
        )

    for topic in STORY_TOPICS:
        add(
            rows,
            f"Write a 5 sentence story about {topic}.",
            (
                "Maya opened her laptop before sunrise. "
                "She checked the data, fixed one bug, and restarted the run. "
                "The tiny model learned to answer more clearly with every batch. "
                "By evening, her eval report showed fewer repeated sentences. "
                "She saved the checkpoint and wrote down what to improve next."
            ),
        )

    concise_pairs = [
        ("Answer yes or no: Is water wet?", "Yes."),
        ("Answer yes or no: Is the sky usually green?", "No."),
        ("Answer with one word: opposite of hot.", "cold"),
        ("Answer with one word: plural of child.", "children"),
        ("Complete this sentence with one short phrase: A GPU is used for", "parallel computation."),
    ]
    for instruction, response in concise_pairs:
        add(rows, instruction, response)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small behavior-patch SFT JSONL set.")
    parser.add_argument("--output", default="data/sft/behavior_patch_train.jsonl")
    parser.add_argument("--eval-output", default="data/sft/behavior_patch_eval.jsonl")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_rows = build_rows()
    rows = []
    for _ in range(args.repeat):
        shuffled = base_rows.copy()
        rng.shuffle(shuffled)
        rows.extend(shuffled)

    eval_rows = base_rows[: args.eval_size]
    rng.shuffle(rows)

    output = Path(args.output)
    eval_output = Path(args.eval_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_output.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"train_rows": len(rows), "eval_rows": len(eval_rows)}, indent=2))


if __name__ == "__main__":
    main()
