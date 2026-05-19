#!/usr/bin/env python3
"""Extract training metrics from JSONL logs and render release-ready curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


def _load_event_lines(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "event" in obj:
            events.append(obj)
    return events


def _maybe_final_eval(path: Path | None, step: int, tokens: int) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.search(r"loss=([0-9.]+)\s+perplexity=([0-9.]+)", text)
    if not match:
        return None
    return {
        "event": "eval",
        "step": step,
        "tokens": tokens,
        "loss": float(match.group(1)),
        "perplexity": float(match.group(2)),
        "source": path.name,
    }


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    out: list[float] = []
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        out.append((prefix[idx + 1] - prefix[start]) / (idx + 1 - start))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event",
        "step",
        "tokens",
        "tokens_b",
        "loss",
        "perplexity",
        "lr",
        "tokens_per_sec_window",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {field: row.get(field, "") for field in fields}
            tokens = row.get("tokens")
            clean["tokens_b"] = f"{tokens / 1e9:.6f}" if isinstance(tokens, (int, float)) else ""
            writer.writerow(clean)


def _write_summary(path: Path, train: list[dict[str, Any]], evals: list[dict[str, Any]]) -> None:
    tps_values = [
        row["tokens_per_sec_window"]
        for row in train
        if isinstance(row.get("tokens_per_sec_window"), (int, float))
    ]
    steady_tps = [
        row["tokens_per_sec_window"]
        for row in train
        if isinstance(row.get("tokens_per_sec_window"), (int, float)) and row.get("step", 0) >= 1000
    ]
    summary = {
        "train_points": len(train),
        "eval_points": len(evals),
        "first_train": train[0] if train else None,
        "last_train": train[-1] if train else None,
        "first_eval": evals[0] if evals else None,
        "last_eval": evals[-1] if evals else None,
        "mean_tokens_per_sec_window": mean(tps_values) if tps_values else None,
        "mean_tokens_per_sec_after_step_1000": mean(steady_tps) if steady_tps else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def _plot_curves(path: Path, train: list[dict[str, Any]], evals: list[dict[str, Any]], smooth: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_x = [row["tokens"] / 1e9 for row in train]
    train_loss = [row["loss"] for row in train]
    train_smooth = _moving_average(train_loss, smooth)
    eval_x = [row.get("tokens", row["step"] * 528384) / 1e9 for row in evals]
    eval_loss = [row["loss"] for row in evals]
    lr = [row.get("lr", math.nan) for row in train]
    tps = [row.get("tokens_per_sec_window", math.nan) for row in train]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(train_x, train_loss, color="#9aa6b2", linewidth=0.7, alpha=0.35, label="train loss")
    axes[0].plot(train_x, train_smooth, color="#1f77b4", linewidth=2.0, label=f"train loss MA-{smooth}")
    axes[0].plot(eval_x, eval_loss, color="#d62728", marker="o", markersize=4, linewidth=1.5, label="validation loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("l20-edu-135m Training Curves")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(train_x, lr, color="#2ca02c", linewidth=1.5)
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(train_x, tps, color="#9467bd", linewidth=1.1)
    axes[2].set_ylabel("Tokens/sec")
    axes[2].set_xlabel("Training tokens (billions)")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_loss_zoom(path: Path, train: list[dict[str, Any]], evals: list[dict[str, Any]], smooth: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_x = [row["tokens"] / 1e9 for row in train]
    train_loss = [row["loss"] for row in train]
    train_smooth = _moving_average(train_loss, smooth)
    eval_x = [row.get("tokens", row["step"] * 528384) / 1e9 for row in evals]
    eval_loss = [row["loss"] for row in evals]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(train_x, train_loss, color="#9aa6b2", linewidth=0.7, alpha=0.25, label="train loss")
    ax.plot(train_x, train_smooth, color="#1f77b4", linewidth=2.0, label=f"train loss MA-{smooth}")
    ax.plot(eval_x, eval_loss, color="#d62728", marker="o", markersize=4, linewidth=1.5, label="validation loss")
    ax.set_xlim(1.0, max(train_x[-1], eval_x[-1]))
    ax.set_ylim(2.65, 3.45)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Loss")
    ax.set_title("l20-edu-135m Loss After Warmup")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--final-eval-log", type=Path)
    parser.add_argument("--final-step", type=int, default=18928)
    parser.add_argument("--final-tokens", type=int, default=10_001_252_352)
    parser.add_argument("--csv", type=Path, default=Path("docs/training_metrics.csv"))
    parser.add_argument("--summary", type=Path, default=Path("docs/training_summary.json"))
    parser.add_argument("--plot", type=Path, default=Path("docs/assets/training_curves.png"))
    parser.add_argument("--loss-plot", type=Path, default=Path("docs/assets/loss_curve_zoom.png"))
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()

    events = _load_event_lines(args.log)
    train = [row for row in events if row.get("event") == "train"]
    evals = [row for row in events if row.get("event") == "eval"]
    final_eval = _maybe_final_eval(args.final_eval_log, args.final_step, args.final_tokens)
    if final_eval and not any(row.get("step") == args.final_step for row in evals):
        evals.append(final_eval)
    train.sort(key=lambda row: row.get("step", 0))
    evals.sort(key=lambda row: row.get("step", 0))

    token_by_step = {row.get("step"): row.get("tokens") for row in train}
    for row in evals:
        row.setdefault("tokens", token_by_step.get(row.get("step"), row.get("step", 0) * 528384))

    rows = train + evals
    rows.sort(key=lambda row: (row.get("step", 0), row.get("event") != "train"))
    _write_csv(args.csv, rows)
    _write_summary(args.summary, train, evals)
    _plot_curves(args.plot, train, evals, args.smooth)
    _plot_loss_zoom(args.loss_plot, train, evals, args.smooth)

    print(f"train_points={len(train)} eval_points={len(evals)}")
    print(f"wrote {args.csv}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.plot}")
    print(f"wrote {args.loss_plot}")


if __name__ == "__main__":
    main()
