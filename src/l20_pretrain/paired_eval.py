"""Paired sample-level statistics for lm-evaluation-harness outputs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "boolq": "acc",
    "hellaswag": "acc_norm",
    "lambada_openai": "acc",
    "openbookqa": "acc_norm",
    "piqa": "acc_norm",
    "sciq": "acc_norm",
    "swag": "acc_norm",
    "winogrande": "acc",
}

LEGACY_TASKS = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
)

INDEPENDENT_DEV_TASKS = (
    "openbookqa",
    "sciq",
    "boolq",
    "swag",
)


@dataclass(frozen=True)
class SampleOutcome:
    correct: float
    doc_hash: str | None


def find_sample_file(root: Path, task: str) -> Path:
    """Return the single sample dump for ``task`` below ``root``."""
    if root.is_file():
        matches = [root] if root.suffix == ".jsonl" else []
    else:
        matches = sorted(root.rglob(f"samples_{task}_*.jsonl"))
        matches += sorted(root.rglob(f"samples_{task}.jsonl"))
        matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one sample file for {task} below {root}, found {len(matches)}"
        )
    return matches[0]


def load_outcomes(path: Path, metric: str) -> dict[str, SampleOutcome]:
    """Load per-document correctness from an lm-eval JSONL sample dump."""
    outcomes: dict[str, SampleOutcome] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "doc_id" not in row:
                raise ValueError(f"{path}:{line_number}: missing doc_id")
            if metric not in row:
                raise ValueError(f"{path}:{line_number}: missing metric {metric}")
            key = str(row["doc_id"])
            if key in outcomes:
                raise ValueError(f"{path}:{line_number}: duplicate doc_id {key}")
            value = float(row[metric])
            if value not in {0.0, 1.0}:
                raise ValueError(
                    f"{path}:{line_number}: {metric} must be binary for paired analysis, got {value}"
                )
            outcomes[key] = SampleOutcome(correct=value, doc_hash=row.get("doc_hash"))
    if not outcomes:
        raise ValueError(f"no samples found in {path}")
    return outcomes


def _paired_deltas(
    baseline: dict[str, SampleOutcome], candidate: dict[str, SampleOutcome]
) -> np.ndarray:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    if baseline_keys != candidate_keys:
        missing_candidate = sorted(baseline_keys - candidate_keys)[:5]
        missing_baseline = sorted(candidate_keys - baseline_keys)[:5]
        raise ValueError(
            "sample key mismatch: "
            f"missing from candidate={missing_candidate}, missing from baseline={missing_baseline}"
        )

    deltas: list[float] = []
    for key in sorted(baseline_keys):
        before = baseline[key]
        after = candidate[key]
        if before.doc_hash and after.doc_hash and before.doc_hash != after.doc_hash:
            raise ValueError(f"doc_hash mismatch for doc_id {key}")
        deltas.append(after.correct - before.correct)
    return np.asarray(deltas, dtype=np.float64)


def _bootstrap_means(
    values: np.ndarray, *, samples: int, rng: np.random.Generator, chunk_size: int = 512
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    result = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        result[start:stop] = values[indices].mean(axis=1)
    return result


def mcnemar_exact_p_value(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value from discordant outcome counts."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    k = min(wins, losses)
    log_probability_at_k = (
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
    )
    relative_sum = 1.0
    relative_term = 1.0
    for i in range(k, 0, -1):
        relative_term *= i / (discordant - i + 1)
        relative_sum += relative_term
    p_value = 2.0 * math.exp(log_probability_at_k + math.log(relative_sum))
    return min(1.0, p_value)


def _interval(values: np.ndarray, confidence: float) -> list[float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return [float(low), float(high)]


def compare_paired_tasks(
    baseline_root: Path,
    candidate_root: Path,
    tasks: Iterable[str] = LEGACY_TASKS,
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260616,
) -> dict[str, Any]:
    """Compare two checkpoints with paired examples and an equal-task mean."""
    task_names = list(tasks)
    unknown = [task for task in task_names if task not in TASK_METRICS]
    if unknown:
        raise ValueError(f"unsupported tasks: {unknown}")
    if not task_names:
        raise ValueError("at least one task is required")

    rng = np.random.default_rng(seed)
    task_results: dict[str, Any] = {}
    bootstrap_by_task: list[np.ndarray] = []
    pooled_wins = pooled_losses = pooled_ties = 0

    for task in task_names:
        metric = TASK_METRICS[task]
        baseline_path = find_sample_file(baseline_root, task)
        candidate_path = find_sample_file(candidate_root, task)
        baseline = load_outcomes(baseline_path, metric)
        candidate = load_outcomes(candidate_path, metric)
        deltas = _paired_deltas(baseline, candidate)
        bootstrap = _bootstrap_means(deltas, samples=bootstrap_samples, rng=rng)
        wins = int(np.count_nonzero(deltas == 1.0))
        losses = int(np.count_nonzero(deltas == -1.0))
        ties = int(np.count_nonzero(deltas == 0.0))
        pooled_wins += wins
        pooled_losses += losses
        pooled_ties += ties
        bootstrap_by_task.append(bootstrap)
        baseline_mean = float(np.mean([row.correct for row in baseline.values()]))
        candidate_mean = float(np.mean([row.correct for row in candidate.values()]))
        task_results[task] = {
            "metric": metric,
            "examples": len(deltas),
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "delta": candidate_mean - baseline_mean,
            "paired_bootstrap_ci": _interval(bootstrap, confidence),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mcnemar_exact_two_sided_p": mcnemar_exact_p_value(wins, losses),
            "baseline_samples": str(baseline_path),
            "candidate_samples": str(candidate_path),
        }

    mean_bootstrap = np.mean(np.stack(bootstrap_by_task, axis=0), axis=0)
    baseline_mean = float(np.mean([row["baseline_mean"] for row in task_results.values()]))
    candidate_mean = float(np.mean([row["candidate_mean"] for row in task_results.values()]))
    return {
        "status": "complete",
        "method": {
            "pairing_key": "task + doc_id; doc_hash equality enforced when present",
            "bootstrap": "paired resampling within each task, then equal-task averaging",
            "bootstrap_samples": bootstrap_samples,
            "confidence": confidence,
            "seed": seed,
            "mcnemar": "two-sided exact binomial test on discordant pairs",
        },
        "tasks": task_results,
        "aggregate": {
            "task_count": len(task_results),
            "baseline_equal_task_mean": baseline_mean,
            "candidate_equal_task_mean": candidate_mean,
            "delta": candidate_mean - baseline_mean,
            "paired_bootstrap_ci": _interval(mean_bootstrap, confidence),
            "pooled_wins": pooled_wins,
            "pooled_losses": pooled_losses,
            "pooled_ties": pooled_ties,
            "pooled_mcnemar_exact_two_sided_p": mcnemar_exact_p_value(
                pooled_wins, pooled_losses
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired sample-level comparison of two lm-eval runs."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    payload = compare_paired_tasks(
        args.baseline,
        args.candidate,
        tasks=args.tasks or LEGACY_TASKS,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
