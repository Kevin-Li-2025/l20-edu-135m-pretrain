from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(".")
OUT = ROOT / "docs" / "project_report"
TASKS = [
    ("arc_challenge", "ARC-Challenge", "acc_norm,none", 0.25),
    ("arc_easy", "ARC-Easy", "acc_norm,none", 0.25),
    ("hellaswag", "HellaSwag", "acc_norm,none", 0.25),
    ("lambada_openai", "LAMBADA OpenAI", "acc,none", 0.0),
    ("piqa", "PIQA", "acc_norm,none", 0.5),
    ("winogrande", "WinoGrande", "acc,none", 0.5),
]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def metric_from_lm_eval(path: Path) -> dict[str, float]:
    data = read_json(path, {})
    out: dict[str, float] = {}
    results = data.get("results", {})
    for task, _, metric, _ in TASKS:
        value = results.get(task, {}).get(metric)
        if value is not None:
            out[task] = float(value)
    return out


def mean_score(scores: dict[str, float]) -> float:
    return sum(scores[t] for t, *_ in TASKS if t in scores) / max(1, sum(1 for t, *_ in TASKS if t in scores))


def adjusted_mean(scores: dict[str, float]) -> float:
    vals = []
    for task, _, _, random_floor in TASKS:
        if task not in scores:
            continue
        denom = max(1e-9, 1.0 - random_floor)
        vals.append(max(0.0, scores[task] - random_floor) / denom)
    return sum(vals) / max(1, len(vals))


def public_baselines() -> dict[str, dict[str, float]]:
    paths = {
        "GPT-2 Small": "eval_results/gpt2-small/*/results_*.json",
        "OPT-125M": "eval_results/opt-125m/*/results_*.json",
        "Cerebras-GPT-111M": "eval_results/cerebras-gpt-111m/*/results_*.json",
        "Pythia-160M": "eval_results/pythia-160m/*/results_*.json",
        "SmolLM-135M": "eval_results/smollm-135m/*/results_*.json",
        "SmolLM2-135M": "eval_results/smollm2-135m/*/results_*.json",
    }
    out: dict[str, dict[str, float]] = {}
    for name, pattern in paths.items():
        matches = sorted(ROOT.glob(pattern))
        if matches:
            out[name] = metric_from_lm_eval(matches[-1])
    return out


def final_model_scores() -> dict[str, float]:
    summary = read_json(ROOT / "eval_results/stage4_release/sft_eval/summary.json", {})
    selected = "stage4-sft-a0875"
    scores: dict[str, float] = {}
    for row in summary.get("tasks", []):
        task = row.get("task")
        if task and selected in row:
            scores[task] = float(row[selected])
    return scores


def sft_interpolation_rows() -> list[dict[str, Any]]:
    summary = read_json(ROOT / "eval_results/stage4_release/sft_eval/summary.json", {})
    means = summary.get("means", {})
    return [{"variant": k, "six_task_mean": v} for k, v in sorted(means.items(), key=lambda kv: kv[1], reverse=True)]


def speed_benchmarks() -> list[dict[str, Any]]:
    rows = []
    for path in [
        ROOT / "docs/l20_speed_benchmark_2k_20260615.jsonl",
        ROOT / "docs/l20_speed_benchmark_4k.jsonl",
        ROOT / "docs/l20_pretrain_flash_benchmark.jsonl",
        ROOT / "logs/stage4_speed_benchmark_compile.jsonl",
        ROOT / "logs/stage4_speed_benchmark_earlystop.jsonl",
    ]:
        for row in iter_jsonl(path):
            if row.get("event") == "benchmark_result":
                row = dict(row)
                row["source_file"] = str(path)
                rows.append(row)
    rows.sort(key=lambda r: float(r.get("tokens_per_sec") or 0), reverse=True)
    return rows


def training_events() -> list[dict[str, Any]]:
    events = []
    for path, run_group in [
        (ROOT / "logs/cpt10b-curriculum/phase1_2k.log", "cpt10b_phase1_2k"),
        (ROOT / "logs/l20_stage4_hq_crossdedup_train_latest.log", "stage4_cpt_8k"),
        (ROOT / "logs/stage4_sft_train_latest.log", "stage4_sft"),
    ]:
        for row in iter_jsonl(path):
            if row.get("event") in {"train", "eval", "checkpoint", "start"}:
                row = dict(row)
                row["source_file"] = str(path)
                row["run_group"] = run_group
                events.append(row)
    return events


def summarize_training(events: list[dict[str, Any]], run_group: str = "cpt10b_phase1_2k") -> dict[str, Any]:
    events = [r for r in events if r.get("run_group") == run_group]
    train_rows = [r for r in events if r.get("event") == "train" and r.get("tokens_per_sec_window")]
    if not train_rows:
        return {}
    latest = train_rows[-1]
    recent = train_rows[-20:]
    avg_tps = sum(float(r["tokens_per_sec_window"]) for r in recent) / len(recent)
    avg_mfu = sum(float(r.get("mfu_pct") or 0.0) for r in recent) / len(recent)
    max_tps = max(float(r["tokens_per_sec_window"]) for r in train_rows)
    max_mfu = max(float(r.get("mfu_pct") or 0.0) for r in train_rows)
    planned = 10_000_000_000 if run_group == "cpt10b_phase1_2k" else int(latest.get("tokens") or 0)
    done = int(latest.get("tokens") or 0)
    remaining_sec = max(0, planned - done) / max(1.0, avg_tps)
    return {
        "latest_step": latest.get("step"),
        "latest_tokens": done,
        "latest_loss": latest.get("loss"),
        "recent_tokens_per_sec": avg_tps,
        "recent_mfu_pct": avg_mfu,
        "max_tokens_per_sec": max_tps,
        "max_mfu_pct": max_mfu,
        "estimated_remaining_hours_for_10b": remaining_sec / 3600,
    }


def energy_summary(events: list[dict[str, Any]], run_group: str = "cpt10b_phase1_2k") -> dict[str, Any]:
    events = [r for r in events if r.get("run_group") == run_group]
    train_rows = [r for r in events if r.get("event") == "train" and r.get("step_time_sec_window")]
    if not train_rows:
        return {}
    # Logs do not contain continuous power samples. Use the observed steady L20 draw from nvidia-smi checks.
    assumed_power_w = 350.0
    elapsed_sec = 0.0
    prev_tokens = 0
    for row in train_rows:
        tokens = int(row.get("tokens") or prev_tokens)
        delta = max(0, tokens - prev_tokens)
        tps = float(row.get("tokens_per_sec_window") or 0.0)
        if delta and tps:
            elapsed_sec += delta / tps
        prev_tokens = tokens
    tokens = max(int(r.get("tokens") or 0) for r in train_rows)
    kwh = assumed_power_w * elapsed_sec / 3_600_000
    return {
        "assumed_gpu_power_w": assumed_power_w,
        "logged_window_seconds": elapsed_sec,
        "logged_tokens": tokens,
        "estimated_gpu_kwh_for_logged_windows": kwh,
        "estimated_gpu_wh_per_million_tokens": (kwh * 1000) / max(1, tokens / 1_000_000),
        "note": "Energy uses observed steady nvidia-smi power around 341-354 W; wall power is higher.",
    }


def data_guard_summary() -> dict[str, Any]:
    meta = read_json(ROOT / "data/l20_stage4_hq_crossdedup_8k/metadata.json", {})
    gate = read_json(ROOT / "eval_results/stage4_release/data_gate.json", {})
    return {
        "train_tokens": meta.get("train_tokens"),
        "val_tokens": meta.get("val_tokens"),
        "source_tokens": meta.get("source_tokens", {}),
        "counters": meta.get("counters", {}),
        "data_guard": meta.get("data_guard", {}),
        "release_gate": gate,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_svg_loss(path: Path, events: list[dict[str, Any]]) -> None:
    train = [(float(r["tokens"]) / 1e9, float(r["loss"])) for r in events if r.get("event") == "train" and r.get("tokens") and r.get("loss")]
    evals = [(float(r.get("step", 0)), float(r["loss"])) for r in events if r.get("event") == "eval" and r.get("loss")]
    if not train:
        return
    w, h = 900, 420
    margin = 50
    xs = [x for x, _ in train]
    ys = [y for _, y in train]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(ymin, ymax):
        ymin -= 0.1
        ymax += 0.1
    def sx(x: float) -> float:
        return margin + (x - xmin) / max(1e-9, xmax - xmin) * (w - 2 * margin)
    def sy(y: float) -> float:
        return h - margin - (y - ymin) / max(1e-9, ymax - ymin) * (h - 2 * margin)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in train)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{h-margin}" x2="{w-margin}" y2="{h-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{h-margin}" stroke="#333"/>',
        f'<polyline fill="none" stroke="#0b6" stroke-width="2" points="{pts}"/>',
        f'<text x="{w/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Training loss curve</text>',
        f'<text x="{w/2}" y="{h-10}" text-anchor="middle" font-family="sans-serif" font-size="13">Billion tokens</text>',
        f'<text x="18" y="{h/2}" transform="rotate(-90 18,{h/2})" text-anchor="middle" font-family="sans-serif" font-size="13">Loss</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines))


def make_report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = final_model_scores()
    baselines = public_baselines()
    events = training_events()
    train_summary = summarize_training(events, "cpt10b_phase1_2k")
    energy = energy_summary(events, "cpt10b_phase1_2k")
    guard = data_guard_summary()
    speed = speed_benchmarks()
    sft_rows = sft_interpolation_rows()

    comparison_rows = []
    comparison_rows.append({
        "model": "L20-edu-135M final a0875",
        "six_task_mean": mean_score(scores),
        "random_adjusted_mean": adjusted_mean(scores),
        **{task: scores.get(task) for task, *_ in TASKS},
    })
    for name, sc in baselines.items():
        comparison_rows.append({
            "model": name,
            "six_task_mean": mean_score(sc),
            "random_adjusted_mean": adjusted_mean(sc),
            **{task: sc.get(task) for task, *_ in TASKS},
        })
    comparison_rows.sort(key=lambda r: float(r["six_task_mean"]), reverse=True)

    write_csv(OUT / "benchmark_comparison.csv", comparison_rows)
    write_csv(OUT / "speed_ablation.csv", speed)
    write_csv(OUT / "sft_interpolation_ablation.csv", sft_rows)
    write_csv(OUT / "training_events.csv", events)
    (OUT / "compute_energy_summary.json").write_text(json.dumps({"training": train_summary, "energy": energy}, indent=2))
    (OUT / "data_quality_summary.json").write_text(json.dumps(guard, indent=2))
    write_svg_loss(OUT / "training_loss.svg", events)

    best_speed = speed[0] if speed else {}
    md = []
    md.append("# L20-edu-135M Technical Report\n")
    md.append("## Current Status\n")
    md.append(f"- Current 10B run tokens: {train_summary.get('latest_tokens', 0):,}\n")
    md.append(f"- Recent throughput: {train_summary.get('recent_tokens_per_sec', 0):,.0f} tok/s\n")
    md.append(f"- Recent MFU: {train_summary.get('recent_mfu_pct', 0):.2f}%\n")
    md.append(f"- Estimated remaining time to 10B: {train_summary.get('estimated_remaining_hours_for_10b', 0):.1f} hours\n")
    md.append("\n## Six-Task Comparison\n")
    md.append("| Model | Mean | Random-adjusted mean | ARC-C | ARC-E | HellaSwag | LAMBADA | PIQA | WinoGrande |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in comparison_rows:
        md.append(
            f"| {row['model']} | {row['six_task_mean']:.4f} | {row['random_adjusted_mean']:.4f} | "
            f"{row.get('arc_challenge', 0):.4f} | {row.get('arc_easy', 0):.4f} | {row.get('hellaswag', 0):.4f} | "
            f"{row.get('lambada_openai', 0):.4f} | {row.get('piqa', 0):.4f} | {row.get('winogrande', 0):.4f} |\n"
        )
    md.append("\n## What Is Already Supported By Evidence\n")
    md.append("- Speed design: 2K context + Liger + compile + micro-batch 16 is the measured fastest training setup.\n")
    if best_speed:
        md.append(f"  Best benchmark: `{best_speed.get('variant')}` at {float(best_speed.get('tokens_per_sec', 0)):,.0f} tok/s.\n")
    md.append("- Data quality: Stage4 metadata records cross-source near-duplicate removal, segment deduplication, 13-gram contamination checks, and LCS filtering.\n")
    md.append("- SFT design: interpolation ablation selected `stage4-sft-a0875`; full SFT was worse than the fused checkpoint.\n")
    md.append("- Efficiency: current run sustains about 50% MFU on a single L20.\n")
    md.append("\n## Evidence Gaps To Close\n")
    md.append("- Per-source data ablation requires additional controlled short runs; current evidence is correlational for the mixture weights.\n")
    md.append("- Long-context curriculum contribution needs checkpoints at 2K-only, 4K, and 8K evaluated under the same six-task harness.\n")
    md.append("- Data cleaning contribution needs a small no-crossdedup or relaxed-filter control, capped to avoid contamination claims.\n")
    md.append("- Statistical robustness needs at least one repeat seed or bootstrap confidence intervals for benchmark deltas.\n")
    md.append("\n## Generated Artifacts\n")
    md.append("- `benchmark_comparison.csv`\n")
    md.append("- `speed_ablation.csv`\n")
    md.append("- `sft_interpolation_ablation.csv`\n")
    md.append("- `training_events.csv`\n")
    md.append("- `compute_energy_summary.json`\n")
    md.append("- `data_quality_summary.json`\n")
    md.append("- `training_loss.svg`\n")
    (OUT / "TECHNICAL_REPORT.md").write_text("".join(md))


if __name__ == "__main__":
    make_report()
