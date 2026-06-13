#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path


DISPLAY = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "hellaswag": "HellaSwag",
    "lambada_openai": "LAMBADA OpenAI",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(summary: dict, model: str) -> str:
    lines = ["| Task | Metric | Score |", "| --- | --- | ---: |"]
    for row in summary["tasks"]:
        value = row.get(model)
        score = "" if value is None else f"{float(value):.4f}"
        lines.append(f"| {DISPLAY.get(row['task'], row['task'])} | {row['metric']} | {score} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--sft-summary", required=True)
    parser.add_argument("--data-gate", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--sft-data-summary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = load(args.base_summary)
    sft = load(args.sft_summary)
    gate = load(args.data_gate)
    best = load(args.best_checkpoint)
    sft_data = load(args.sft_data_summary) if args.sft_data_summary else {}
    content = f"""---
library_name: transformers
pipeline_tag: text-generation
tags:
- continual-pretraining
- sft
- 135m
---

# L20 Edu 135M Stage 4

The repository root contains the anti-forgetting SFT model. The selected base
checkpoint is archived under `releases/stage4-best`.

## Training

- Parameters: 134,515,008
- Context length: 8,192 tokens for continued pretraining
- Stage 4 data: {gate['train_tokens']:,} tokens
- Cross-source deduplication: 64-permutation MinHash with LSH candidate search
- Benchmark decontamination: 13-gram candidate matching plus token LCS >= 0.60
- Selected base checkpoint: step {best['step']}
- Selected validation loss: {best.get('eval_loss')}
- SFT data: HuggingFaceTB/smol-smoltalk, filtered for the 135M model
- SFT training rows: {int(sft_data.get('train_rows', 0)):,}
- SFT decontamination: 13-gram candidates plus token LCS >= 0.60
- Anti-forgetting: base/SFT interpolation candidates selected by benchmark regression gates

## Stage 4 Base Results

{table(base, 'stage4-best')}

## SFT Results

{table(sft, 'stage4-sft')}

## Data Gate

- Status: `{gate['status']}`
- Validation tokens: {gate['val_tokens']:,}
- Benchmark-contaminated documents removed: {gate['contamination_rejections']:,}
- Indexed documents: {gate['dedup_index_counts'].get('documents', 0):,}
- Indexed sentence/paragraph segments: {gate['dedup_index_counts'].get('segments', 0):,}

## Reproducibility

Evaluation uses `lm-evaluation-harness` with a fixed seed and the tasks
ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI, PIQA, and WinoGrande.
Artifacts and summaries are stored under `eval_results/`.

Generated: {datetime.now(UTC).isoformat()}

## Limitations

This is a small research model. Benchmark results do not establish general
state of the art, safety, factual reliability, or suitability for high-stakes
use.
"""
    Path(args.out).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
