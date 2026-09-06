#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from l20_pretrain.config import PretrainConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a matched pretraining pilot pair.")
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--schedule-config", type=Path)
    parser.add_argument("--schedule-log", type=Path)
    parser.add_argument("--baseline-schedule-config", type=Path)
    parser.add_argument("--baseline-schedule-log", type=Path)
    parser.add_argument("--data-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("event"), str):
            events.append(payload)
    return events


def assert_matched(candidate: PretrainConfig, baseline: PretrainConfig) -> None:
    fields = {
        "seed": (candidate.seed, baseline.seed),
        "tokenizer": (candidate.tokenizer_name, baseline.tokenizer_name),
        "dataset path": (candidate.dataset.tokenized_path, baseline.dataset.tokenized_path),
        "dataset split": (candidate.dataset.split, baseline.dataset.split),
        "block size": (candidate.model.block_size, baseline.model.block_size),
        "micro batch": (
            candidate.trainer.micro_batch_size,
            baseline.trainer.micro_batch_size,
        ),
        "gradient accumulation": (
            candidate.trainer.gradient_accumulation_steps,
            baseline.trainer.gradient_accumulation_steps,
        ),
        "max steps": (candidate.trainer.max_steps, baseline.trainer.max_steps),
        "warmup steps": (candidate.trainer.warmup_steps, baseline.trainer.warmup_steps),
        "learning rate": (
            candidate.trainer.learning_rate,
            baseline.trainer.learning_rate,
        ),
        "minimum LR ratio": (
            candidate.trainer.min_lr_ratio,
            baseline.trainer.min_lr_ratio,
        ),
        "weight decay": (
            candidate.trainer.weight_decay,
            baseline.trainer.weight_decay,
        ),
        "dtype": (candidate.trainer.dtype, baseline.trainer.dtype),
        "planned tokens": (candidate.planned_tokens, baseline.planned_tokens),
        "eval batches": (candidate.trainer.eval_batches, baseline.trainer.eval_batches),
    }
    mismatches = [f"{name}: {left!r} != {right!r}" for name, (left, right) in fields.items() if left != right]
    if mismatches:
        raise ValueError("pilot pair is not matched:\n" + "\n".join(mismatches))
    if candidate.dataset.allow_repetition or baseline.dataset.allow_repetition:
        raise ValueError("matched one-pass pilots must set dataset.allow_repetition=false")


def assert_schedule_ablation(candidate: PretrainConfig, schedule: PretrainConfig) -> None:
    fields = {
        "seed": (candidate.seed, schedule.seed),
        "tokenizer": (candidate.tokenizer_name, schedule.tokenizer_name),
        "dataset path": (candidate.dataset.tokenized_path, schedule.dataset.tokenized_path),
        "model": (candidate.model, schedule.model),
        "micro batch": (candidate.trainer.micro_batch_size, schedule.trainer.micro_batch_size),
        "gradient accumulation": (
            candidate.trainer.gradient_accumulation_steps,
            schedule.trainer.gradient_accumulation_steps,
        ),
        "max steps": (candidate.trainer.max_steps, schedule.trainer.max_steps),
        "warmup steps": (candidate.trainer.warmup_steps, schedule.trainer.warmup_steps),
        "learning rate": (candidate.trainer.learning_rate, schedule.trainer.learning_rate),
        "weight decay": (candidate.trainer.weight_decay, schedule.trainer.weight_decay),
        "dtype": (candidate.trainer.dtype, schedule.trainer.dtype),
        "planned tokens": (candidate.planned_tokens, schedule.planned_tokens),
        "eval batches": (candidate.trainer.eval_batches, schedule.trainer.eval_batches),
    }
    mismatches = [f"{name}: {left!r} != {right!r}" for name, (left, right) in fields.items() if left != right]
    if mismatches:
        raise ValueError("schedule ablation changes non-schedule controls:\n" + "\n".join(mismatches))
    if candidate.trainer.lr_schedule != "cosine" or schedule.trainer.lr_schedule != "wsd":
        raise ValueError("schedule ablation must compare cosine candidate against wsd")
    if candidate.dataset.allow_repetition or schedule.dataset.allow_repetition:
        raise ValueError("schedule ablation must be one-pass")


def summarize_run(config: PretrainConfig, config_path: Path, log_path: Path) -> dict[str, Any]:
    events = json_events(log_path)
    starts = [event for event in events if event["event"] == "start"]
    trains = [event for event in events if event["event"] == "train"]
    evals = [event for event in events if event["event"] == "eval"]
    dones = [event for event in events if event["event"] == "done"]
    if len(starts) != 1 or not trains or len(evals) != 1 or len(dones) != 1:
        raise RuntimeError(
            f"incomplete run evidence in {log_path}: "
            f"starts={len(starts)}, trains={len(trains)}, evals={len(evals)}, dones={len(dones)}"
        )
    if int(evals[0]["step"]) != config.trainer.max_steps:
        raise RuntimeError(f"final eval step does not match max_steps in {log_path}")
    checkpoint = Path(str(dones[0]["checkpoint"]))
    model_path = checkpoint / "model.safetensors"
    trainer_state_path = checkpoint / "trainer_state.pt"
    throughput = [float(event["tokens_per_sec_window"]) for event in trains[1:]] or [float(trains[0]["tokens_per_sec_window"])]
    mfus = [float(event["mfu_pct"]) for event in trains[1:] if "mfu_pct" in event]
    return {
        "run_name": config.run_name,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "parameters": int(starts[0]["parameters"]),
        "planned_tokens": config.planned_tokens,
        "completed_train_tokens": config.planned_tokens,
        "last_logged_tokens": int(trains[-1]["tokens"]),
        "first_logged_train_loss": float(trains[0]["loss"]),
        "last_logged_train_loss": float(trains[-1]["loss"]),
        "final_eval_loss": float(evals[0]["loss"]),
        "final_eval_perplexity": float(evals[0]["perplexity"]),
        "median_tokens_per_sec": median(throughput),
        "median_mfu_pct": median(mfus) if mfus else None,
        "checkpoint": str(checkpoint),
        "model_sha256": sha256(model_path),
        "trainer_state_sha256": sha256(trainer_state_path),
    }


def main() -> None:
    args = parse_args()
    candidate = load_config(args.candidate_config)
    baseline = load_config(args.baseline_config)
    assert_matched(candidate, baseline)
    metadata = json.loads(args.data_metadata.read_text(encoding="utf-8"))
    candidate_result = summarize_run(candidate, args.candidate_config, args.candidate_log)
    baseline_result = summarize_run(baseline, args.baseline_config, args.baseline_log)
    candidate_loss = candidate_result["final_eval_loss"]
    baseline_loss = baseline_result["final_eval_loss"]
    result = {
        "schema_version": 2,
        "scope": "matched_50m_pilot_not_final_quality_evidence",
        "matched_controls": {
            "seed": candidate.seed,
            "tokenizer": candidate.tokenizer_name,
            "dataset_revision": metadata.get("dataset_revision"),
            "train_artifact": (metadata.get("artifacts") or {}).get("train.bin"),
            "val_artifact": (metadata.get("artifacts") or {}).get("val.bin"),
            "packed_train_blocks": int(metadata["train_tokens"]) // candidate.model.block_size,
            "consumed_train_blocks": (
                candidate.trainer.max_steps
                * candidate.trainer.micro_batch_size
                * candidate.trainer.gradient_accumulation_steps
            ),
            "planned_tokens": candidate.planned_tokens,
        },
        "candidate": candidate_result,
        "baseline": baseline_result,
        "comparison": {
            "lower_eval_loss": (
                candidate.run_name if candidate_loss < baseline_loss else baseline.run_name
            ),
            "candidate_minus_baseline_eval_loss": candidate_loss - baseline_loss,
        },
    }
    if (args.schedule_config is None) != (args.schedule_log is None):
        raise ValueError("--schedule-config and --schedule-log must be provided together")
    if args.schedule_config is not None and args.schedule_log is not None:
        schedule = load_config(args.schedule_config)
        assert_schedule_ablation(candidate, schedule)
        schedule_result = summarize_run(schedule, args.schedule_config, args.schedule_log)
        schedule_loss = schedule_result["final_eval_loss"]
        result["schedule_ablation"] = schedule_result
        result["schedule_comparison"] = {
            "lower_eval_loss": (
                candidate.run_name if candidate_loss < schedule_loss else schedule.run_name
            ),
            "wsd_minus_cosine_eval_loss": schedule_loss - candidate_loss,
        }
    if (args.baseline_schedule_config is None) != (args.baseline_schedule_log is None):
        raise ValueError(
            "--baseline-schedule-config and --baseline-schedule-log must be provided together"
        )
    if args.baseline_schedule_config is not None and args.baseline_schedule_log is not None:
        baseline_schedule = load_config(args.baseline_schedule_config)
        assert_schedule_ablation(baseline, baseline_schedule)
        baseline_schedule_result = summarize_run(
            baseline_schedule,
            args.baseline_schedule_config,
            args.baseline_schedule_log,
        )
        baseline_schedule_loss = baseline_schedule_result["final_eval_loss"]
        result["baseline_schedule_ablation"] = baseline_schedule_result
        result["baseline_schedule_comparison"] = {
            "lower_eval_loss": (
                baseline.run_name
                if baseline_loss < baseline_schedule_loss
                else baseline_schedule.run_name
            ),
            "wsd_minus_cosine_eval_loss": baseline_schedule_loss - baseline_loss,
        }

    run_keys = [
        key
        for key in (
            "candidate",
            "baseline",
            "schedule_ablation",
            "baseline_schedule_ablation",
        )
        if key in result
    ]
    winner_key = min(run_keys, key=lambda key: result[key]["final_eval_loss"])
    result["global_comparison"] = {
        "completed_cells": len(run_keys),
        "lowest_eval_loss": result[winner_key]["final_eval_loss"],
        "winner": result[winner_key]["run_name"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
