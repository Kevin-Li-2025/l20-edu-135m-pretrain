#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_METRICS = {
    "arc_challenge": ("ARC-Challenge", "acc_norm,none"),
    "arc_easy": ("ARC-Easy", "acc_norm,none"),
    "hellaswag": ("HellaSwag", "acc_norm,none"),
    "lambada_openai": ("LAMBADA OpenAI", "acc,none"),
    "piqa": ("PIQA", "acc_norm,none"),
    "winogrande": ("WinoGrande", "acc,none"),
}

PROFILE_PATTERNS = (
    "sm__pipe_tensor",
    "tensor",
    "hmma",
    "mma",
    "roofline",
    "speedoflight",
    "compute throughput",
    "gpu speed of light",
    "nvgpuctrperm",
    "performance counters",
    "permission",
)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def summarize_train_log(path: Path) -> dict[str, Any]:
    events = read_json_lines(path)
    train_events = [event for event in events if event.get("event") == "train"]
    eval_events = [event for event in events if event.get("event") == "eval"]
    checkpoint_events = [event for event in events if event.get("event") == "checkpoint"]
    latest_train = train_events[-1] if train_events else None
    latest_eval = eval_events[-1] if eval_events else None

    mfu_values = [
        float(event["mfu_pct"])
        for event in train_events
        if isinstance(event.get("mfu_pct"), int | float)
    ]
    tok_values = [
        float(event["tokens_per_sec_window"])
        for event in train_events
        if isinstance(event.get("tokens_per_sec_window"), int | float)
    ]
    stable_mfu_values = [
        float(event["mfu_pct"])
        for event in train_events
        if isinstance(event.get("step"), int)
        and int(event["step"]) >= 100
        and isinstance(event.get("mfu_pct"), int | float)
    ]

    return {
        "log": str(path),
        "latest_train": latest_train,
        "latest_eval": latest_eval,
        "eval_history": eval_events,
        "checkpoint_history": checkpoint_events,
        "train_event_count": len(train_events),
        "mfu_pct_mean": mean(mfu_values),
        "mfu_pct_max": max(mfu_values) if mfu_values else None,
        "mfu_pct_stable_mean": mean(stable_mfu_values),
        "tokens_per_sec_mean": mean(tok_values),
        "tokens_per_sec_max": max(tok_values) if tok_values else None,
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def find_result_json(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.rglob("results_*.json")) + sorted(path.rglob("*.json"))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            return candidate
    return None


def summarize_eval(path: Path) -> dict[str, Any]:
    result_json = find_result_json(path)
    if result_json is None:
        return {"status": "missing", "path": str(path), "tasks": {}, "mean": None}
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    tasks: dict[str, Any] = {}
    values: list[float] = []
    for task, (display, metric) in TASK_METRICS.items():
        value = None
        if isinstance(results.get(task), dict):
            raw = results[task].get(metric)
            if isinstance(raw, int | float):
                value = float(raw)
        tasks[task] = {"display": display, "metric": metric, "value": value}
        if value is not None and math.isfinite(value):
            values.append(value)
    return {
        "status": "complete" if len(values) == len(TASK_METRICS) else "partial",
        "path": str(path),
        "result_json": str(result_json),
        "tasks": tasks,
        "mean": mean(values),
    }


def latest_profile_dir(profile_root: Path) -> Path | None:
    if not profile_root.exists():
        return None
    candidates = sorted(
        (path for path in profile_root.glob("tensor_profile_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    return candidates[-1] if candidates else None


def grep_profile_text(path: Path, limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = raw.strip()
            lowered = text.lower()
            if any(pattern in lowered for pattern in PROFILE_PATTERNS):
                lines.append(text)
            if len(lines) >= limit:
                break
    return lines


def profile_csv_rows(path: Path, limit: int = 80) -> list[list[str]]:
    if not path.exists():
        return []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            lowered = " ".join(row).lower()
            if any(pattern in lowered for pattern in PROFILE_PATTERNS):
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def summarize_profile(profile_root: Path) -> dict[str, Any]:
    run_dir = latest_profile_dir(profile_root)
    if run_dir is None:
        return {"status": "missing", "profile_root": str(profile_root)}
    status_path = run_dir / "status.json"
    status = None
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = None
    return {
        "status": "complete"
        if isinstance(status, dict) and status.get("event") == "ncu_profile_done"
        else "failed"
        if isinstance(status, dict) and status.get("event") == "ncu_profile_failed"
        else "present",
        "run_dir": str(run_dir),
        "status_json": status,
        "report": str(run_dir / "stage6_tensor_profile.ncu-rep"),
        "details_snippets": grep_profile_text(run_dir / "ncu_details.txt"),
        "raw_csv_rows": profile_csv_rows(run_dir / "ncu_raw.csv"),
        "log_snippets": grep_profile_text(run_dir / "ncu_profile.log", limit=40),
    }


def fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    if value is None:
        return ""
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    train = summary["train"]
    latest_train = train.get("latest_train") or {}
    latest_eval = train.get("latest_eval") or {}
    eval_summary = summary["eval"]
    profile = summary["profile"]

    lines = [
        "# Stage6 Post-Train Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Training",
        "",
        f"- Latest step: `{latest_train.get('step', '')}`",
        f"- Latest tokens: `{latest_train.get('tokens', '')}`",
        f"- Latest train loss: `{fmt(latest_train.get('loss'))}`",
        f"- Latest tokens/s: `{fmt(latest_train.get('tokens_per_sec_window'))}`",
        f"- Latest MFU: `{fmt(latest_train.get('mfu_pct'))}%`",
        f"- Latest eval loss: `{fmt(latest_eval.get('loss'))}`",
        f"- Latest eval perplexity: `{fmt(latest_eval.get('perplexity'))}`",
        f"- Stable mean MFU: `{fmt(train.get('mfu_pct_stable_mean'))}%`",
        "",
        "## Six-Task Eval",
        "",
        f"- Status: `{eval_summary.get('status')}`",
        f"- Mean: `{fmt(eval_summary.get('mean'))}`",
        "",
        "| Task | Metric | Score |",
        "| --- | --- | --- |",
    ]
    for task in TASK_METRICS:
        item = eval_summary.get("tasks", {}).get(task, {})
        lines.append(
            f"| {item.get('display', task)} | {item.get('metric', '')} | {fmt(item.get('value'))} |"
        )

    lines.extend(
        [
            "",
            "## Nsight Compute Profile",
            "",
            f"- Status: `{profile.get('status')}`",
            f"- Run dir: `{profile.get('run_dir', profile.get('profile_root', ''))}`",
            f"- Report: `{profile.get('report', '')}`",
            "",
            "### Tensor-Core / Roofline Evidence Snippets",
            "",
        ]
    )
    snippets = profile.get("details_snippets") or profile.get("log_snippets") or []
    if snippets:
        lines.extend(f"- `{line[:220]}`" for line in snippets[:20])
    else:
        lines.append("- No Tensor Core or roofline snippets were parsed yet.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage6 training, ncu, and lm-eval outputs.")
    parser.add_argument("--train-log", default="logs/stage6-edu-reasoning/train_stage6.log")
    parser.add_argument("--profile-dir", default="logs/stage6-edu-reasoning/profile")
    parser.add_argument("--eval-dir", default="eval_results/stage6_edu_reasoning_300m")
    parser.add_argument("--out-json", default="results/stage6/posttrain_summary.json")
    parser.add_argument("--out-md", default="results/stage6/posttrain_summary.md")
    args = parser.parse_args()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train": summarize_train_log(Path(args.train_log)),
        "profile": summarize_profile(Path(args.profile_dir)),
        "eval": summarize_eval(Path(args.eval_dir)),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
