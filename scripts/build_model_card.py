#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.config import load_config
from l20_pretrain.eval_results import extract_primary_metrics


def render_eval_table(eval_dir: str | None) -> str:
    if not eval_dir:
        return "| Task | Metric | Score |\n| --- | --- | ---: |\n"
    try:
        metrics = extract_primary_metrics(eval_dir)
    except Exception:
        return "| Task | Metric | Score |\n| --- | --- | ---: |\n"
    lines = ["| Task | Metric | Score |", "| --- | --- | ---: |"]
    for task, metric in sorted(metrics.items()):
        lines.append(f"| {task} | {metric.metric} | {metric.value:.4f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a release model card from config and eval results.")
    parser.add_argument("--config", default="configs/l20_135m_deepthin.yaml")
    parser.add_argument("--checkpoint", default="runs/l20-edu-135m-deepthin/final")
    parser.add_argument("--eval-dir", default="eval_results/l20-edu-135m-deepthin")
    parser.add_argument("--comparison", default="eval_results/comparison.md")
    parser.add_argument("--contamination", default="eval_results/contamination_report.json")
    parser.add_argument("--out", default="MODEL_CARD.md")
    args = parser.parse_args()

    config = load_config(args.config)
    comparison_text = ""
    comparison_path = Path(args.comparison)
    if comparison_path.exists():
        comparison_text = comparison_path.read_text(encoding="utf-8")
    else:
        comparison_text = "Comparison has not been generated yet."

    contamination_path = Path(args.contamination)
    contamination_text = (
        contamination_path.read_text(encoding="utf-8")
        if contamination_path.exists()
        else "Contamination report has not been generated yet."
    )

    content = f"""# {config.run_name}

Generated: {datetime.now(UTC).date().isoformat()}

## Model

- Type: base causal language model
- Architecture: Llama-style decoder-only Transformer
- Shape: {config.model.num_hidden_layers} layers, {config.model.hidden_size} hidden size, {config.model.intermediate_size} intermediate size
- Attention: {config.model.num_attention_heads} query heads, {config.model.num_key_value_heads} KV heads
- Tokenizer: `{config.tokenizer_name}`
- Context length: {config.model.block_size}
- Checkpoint: `{args.checkpoint}`

## Training

- Config: `{args.config}`
- Data: `{config.dataset.name}` / `{config.dataset.config_name}`
- Text filter: min_chars={config.dataset.min_chars}, max_chars={config.dataset.max_chars}, min_score={config.dataset.min_score}, min_int_score={config.dataset.min_int_score}
- Token budget: {config.planned_tokens:,}
- Tokens per optimizer step: {config.tokens_per_step:,}
- Optimizer: AdamW, beta1={config.trainer.beta1}, beta2={config.trainer.beta2}, weight_decay={config.trainer.weight_decay}
- LR schedule: warmup={config.trainer.warmup_steps}, max_lr={config.trainer.learning_rate}, min_lr_ratio={config.trainer.min_lr_ratio}
- Precision: {config.trainer.dtype}

## Evaluation

{render_eval_table(args.eval_dir)}

## Comparison

{comparison_text}

## Contamination

```json
{contamination_text}
```

## Limitations

- This is a base model, not an instruction-tuned assistant.
- It is not safety aligned.
- Benchmark scores are only valid under the exact harness and contamination
  statement used for this release.
- Do not claim general SOTA unless `scripts/sota_gate.py` passes.
"""

    out_path = Path(args.out)
    out_path.write_text(content, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
