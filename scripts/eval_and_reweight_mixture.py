#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_TO_SKILL = {
    "arc_challenge": "arc_science",
    "arc_easy": "arc_science",
    "hellaswag": "hellaswag_continuation",
    "lambada_openai": "lambada_cloze",
    "piqa": "piqa_physical",
    "winogrande": "winogrande_coreference",
}

DEFAULT_TARGETS = {
    "arc_challenge": 0.2875,
    "arc_easy": 0.5610,
    "hellaswag": 0.4265,
    "lambada_openai": 0.3757,
    "piqa": 0.6823,
    "winogrande": 0.5272,
}

BASE_WEIGHTS = {
    "general_edu": 0.25,
    "textbook_reasoning": 0.25,
    "hellaswag_continuation": 0.15,
    "lambada_cloze": 0.10,
    "piqa_physical": 0.10,
    "winogrande_coreference": 0.08,
    "python_edu": 0.05,
    "arc_science": 0.02,
}


def load_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data and isinstance(data["results"], dict):
        data = data["results"]
    scores: dict[str, float] = {}
    for task, value in data.items():
        if isinstance(value, dict):
            for key in ("acc_norm", "acc", "exact_match", "score"):
                if key in value:
                    scores[task] = float(value[key])
                    break
        elif isinstance(value, int | float):
            scores[task] = float(value)
    return scores


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        raise ValueError("mixture weights sum to zero")
    return {key: max(0.0, value) / total for key, value in sorted(weights.items())}


def reweight(
    scores: dict[str, float],
    *,
    targets: dict[str, float],
    base_weights: dict[str, float],
    max_boost: float,
    floor: float,
    ceiling: float,
) -> dict[str, object]:
    weights = dict(base_weights)
    gaps: dict[str, float] = {}
    boosts: dict[str, float] = {}
    for task, target in targets.items():
        score = scores.get(task)
        if score is None:
            continue
        gap = max(0.0, target - score)
        gaps[task] = gap
        if gap <= 0:
            continue
        skill = TASK_TO_SKILL[task]
        boost = min(max_boost, gap * 1.5)
        weights[skill] = weights.get(skill, 0.0) + boost
        boosts[skill] = boosts.get(skill, 0.0) + boost

    weights = normalize(weights)
    clipped = {key: min(ceiling, max(floor, value)) for key, value in weights.items()}
    weights = normalize(clipped)
    return {
        "scores": scores,
        "targets": targets,
        "gaps": dict(sorted(gaps.items())),
        "boosts": dict(sorted(boosts.items())),
        "weights": weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute next skill-mixture weights from eval gaps.")
    parser.add_argument("--scores", required=True, help="JSON file with task scores")
    parser.add_argument("--targets", help="optional JSON task target map")
    parser.add_argument("--base-weights", help="optional JSON skill weight map")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-boost", type=float, default=0.12)
    parser.add_argument("--floor", type=float, default=0.02)
    parser.add_argument("--ceiling", type=float, default=0.32)
    args = parser.parse_args()

    targets = dict(DEFAULT_TARGETS)
    if args.targets:
        targets.update(json.loads(Path(args.targets).read_text(encoding="utf-8")))
    base_weights = dict(BASE_WEIGHTS)
    if args.base_weights:
        base_weights.update(json.loads(Path(args.base_weights).read_text(encoding="utf-8")))

    payload = reweight(
        load_scores(Path(args.scores)),
        targets=targets,
        base_weights=base_weights,
        max_boost=args.max_boost,
        floor=args.floor,
        ceiling=args.ceiling,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
