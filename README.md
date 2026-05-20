# l20-edu-135m-pretrain

From-scratch pretraining of a 134.5M-parameter Llama-style base language model on
10B FineWeb-Edu tokens using a single NVIDIA L20 GPU.

The released checkpoint is available on Hugging Face:
[`AliceYin/l20-edu-135m`](https://huggingface.co/AliceYin/l20-edu-135m).

This project is intentionally scoped as a reproducible small-model pretraining
run, not a general SOTA claim. The useful claim is narrower: a complete
single-GPU pretraining pipeline with public checkpoint, training config,
generation support, perplexity evaluation, and matched `lm-eval` comparisons
against public 100M-160M baselines.

## Result Summary

- Model: `l20-edu-135m`, 134,515,008 parameters
- Architecture: Llama-style decoder-only Transformer
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`
- Dataset: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- Training budget: 10,001,252,352 planned tokens
- Hardware: one NVIDIA L20 GPU
- Final checkpoint: `runs/l20-edu-135m-deepthin/step-018928`
- Final validation: loss `2.8731`, perplexity `17.69`
- Public release: Hugging Face model repo with weights, tokenizer, config,
  training config, model card, and eval comparison files

Final zero-shot `lm-eval` results:

| Task | Metric | Score |
| --- | --- | ---: |
| ARC-Challenge | acc_norm | 0.2765 |
| ARC-Easy | acc_norm | 0.5059 |
| HellaSwag | acc_norm | 0.3272 |
| LAMBADA OpenAI | acc | 0.2540 |
| PIQA | acc_norm | 0.6224 |
| WinoGrande | acc | 0.5099 |

Against public baselines on the same task set, the model beats GPT-2 small on
5/6 tasks, OPT-125M on 4/6, GPT-Neo-125M on 4/6, Cerebras-GPT-111M on 6/6, and
Pythia-160M on 6/6. It does not beat SmolLM-135M or SmolLM2-135M, which were
trained with much larger token budgets.

See [docs/evaluation_report.md](docs/evaluation_report.md) for the full
comparison table, benchmark protocol, contamination status, and training-token
context. See [docs/training_recipe.md](docs/training_recipe.md) for the exact
training recipe.

## Benchmark Rigor

The public baseline comparison uses the same EleutherAI `lm-evaluation-harness`
version (`0.4.12`), task list, zero-shot setting, `bfloat16` dtype, `cuda:0`
device, auto batch policy, full task datasets, logged samples, and comparison
parser for both candidate and baselines. Baseline numbers are self-run through
`scripts/eval_public_baselines.sh`; they are not copied from public leaderboards.

The comparison is still not fully controlled: public baselines use their own
released tokenizers and model context configs. A strict architecture claim
requires the controlled baseline in `configs/l20_wide_140m_baseline.yaml`,
trained with the same tokenizer, FineWeb-Edu slice, context length, optimizer,
schedule, and token budget.

No full contamination pass is claimed for this release. The repository includes
`scripts/check_contamination.py` and `scripts/sample_training_text.py`, but a
separate audit against the benchmark samples is still needed before making a
strong no-contamination statement.

## Training Curves

The run logged 1,903 training points and 38 validation points. The full extracted
metrics are available in [docs/training_metrics.csv](docs/training_metrics.csv),
with a compact summary in [docs/training_summary.json](docs/training_summary.json).

![Loss curve after warmup](docs/assets/loss_curve_zoom.png)

![Training curves](docs/assets/training_curves.png)

## What This Demonstrates

- End-to-end base model pretraining from random initialization.
- Streaming data ingestion and token packing for FineWeb-Edu.
- Checkpointing, resume, generation, validation perplexity, and public eval.
- A documented training recipe: batch size, global batch, sequence length,
  optimizer, LR schedule, warmup, weight decay, gradient accumulation,
  checkpoint cadence, runtime estimate, and known run issues.
- A practical single-GPU recipe for 100M-class models.
- Clear release hygiene: model card, training budget disclosure, baseline
  context, and limitation statements.

## What This Does Not Claim

- This is not a chat model.
- This is not a general SOTA model.
- This is not a matched-token-budget win over SmolLM or SmolLM2.
- This is not evidence that the architecture is better until the controlled
  wide baseline is trained under the same data, tokenizer, optimizer, schedule,
  and token budget.

## Repository Layout

```text
configs/                      Training and evaluation configs
docs/                         Protocols, model card template, eval report
scripts/                      Training, evaluation, comparison, and release tools
src/l20_pretrain/             Model, data, training, generation, eval code
tests/                        Unit tests for config, data, model, eval parsing
```

Large artifacts such as checkpoints, raw eval outputs, logs, and datasets are
not committed to Git. The released model artifacts live on Hugging Face.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install a PyTorch build that matches your CUDA runtime before `pip install -e .`
if the default install is not suitable for the target machine.

## Train

The completed run used:

```bash
python -m l20_pretrain.train configs/l20_135m_deepthin.yaml
```

Resume from a checkpoint:

```bash
python -m l20_pretrain.train configs/l20_135m_deepthin.yaml \
  --resume runs/l20-edu-135m-deepthin/step-010000
```

Generate from a checkpoint:

```bash
python -m l20_pretrain.generate runs/l20-edu-135m-deepthin/step-018928 \
  --prompt "The reason transformers use attention is" \
  --max-new-tokens 120
```

Evaluate perplexity:

```bash
python -m l20_pretrain.eval_ppl \
  runs/l20-edu-135m-deepthin/step-018928 \
  configs/l20_135m_deepthin.yaml
```

Run the base-model eval suite:

```bash
scripts/setup_eval_env.sh
scripts/eval_lm_harness.sh runs/l20-edu-135m-deepthin/step-018928
```

Run public baselines:

```bash
scripts/eval_public_baselines.sh
```

Compare results:

```bash
python scripts/compare_lm_eval.py \
  --candidate l20-edu-135m-deepthin=eval_results/l20-edu-135m-deepthin \
  --baseline gpt2-small=eval_results/gpt2-small \
  --baseline opt-125m=eval_results/opt-125m \
  --baseline gpt-neo-125m=eval_results/gpt-neo-125m \
  --baseline cerebras-gpt-111m=eval_results/cerebras-gpt-111m \
  --baseline pythia-160m=eval_results/pythia-160m \
  --baseline smollm-135m=eval_results/smollm-135m \
  --baseline smollm2-135m=eval_results/smollm2-135m
```

## Supervised Fine-Tuning

The repository includes a runnable SFT scaffold and one completed first-pass
instruction-tuning run. The completed `6k_quality` run is useful evidence for
the post-training pipeline, but it is not publish-quality as a chat assistant:
it lowers held-out SFT loss while still showing repetition and format failures.

- Config: [configs/l20_edu_135m_sft.yaml](configs/l20_edu_135m_sft.yaml)
- Curated-run configs:
  [1k-long](configs/l20_edu_135m_sft_1k_long.yaml),
  [6k-quality](configs/l20_edu_135m_sft_6k_quality.yaml),
  [6k-quality offline](configs/l20_edu_135m_sft_6k_quality_offline.yaml),
  [20k-mixed](configs/l20_edu_135m_sft_20k_mixed.yaml)
- Script: [src/l20_pretrain/train_sft.py](src/l20_pretrain/train_sft.py)
- Data selector: [scripts/prepare_sft_data.py](scripts/prepare_sft_data.py)
- Sanity eval: [scripts/eval_sft_sanity.py](scripts/eval_sft_sanity.py)
- Recipe: [docs/sft_recipe.md](docs/sft_recipe.md)
- HF model card template:
  [README_HF_l20-edu-135m-sft-template.md](README_HF_l20-edu-135m-sft-template.md)

Completed `6k_quality` SFT v1:

| Field | Value |
| --- | --- |
| Base checkpoint | `runs/l20-edu-135m-deepthin/step-018928` |
| Train examples | 6,000 quality-filtered UltraChat rows |
| Eval examples | 512 UltraChat rows |
| Global batch | 64 sequences |
| Max steps | 300 |
| Final checkpoint | `runs/l20-edu-135m-sft-6k-quality/step-000300` |
| Final train loss | 2.0336 |
| Final eval loss / perplexity | 2.0050 / 7.43 |
| Sanity automatic checks | 3 / 5 passed |
| Release verdict | Not publish-quality yet; needs a more conservative follow-up run |

Artifacts:

- Metrics: [docs/sft_6k_quality_metrics.csv](docs/sft_6k_quality_metrics.csv)
- Summary: [docs/sft_6k_quality_summary.json](docs/sft_6k_quality_summary.json)
- Sanity report: [docs/sft_6k_quality_sanity_report.md](docs/sft_6k_quality_sanity_report.md)

![SFT 6k-quality loss curve](docs/assets/sft_6k_quality_loss_curve.png)

Prepare the recommended 6k-quality SFT split:

```bash
python scripts/prepare_sft_data.py \
  --strategy quality \
  --target-size 6000 \
  --eval-size 512 \
  --output data/sft/ultrachat_6k_quality.jsonl \
  --eval-output data/sft/ultrachat_eval_512.jsonl \
  --summary-output data/sft/ultrachat_6k_quality_summary.json
```

Run the main SFT candidate:

```bash
python -m l20_pretrain.train_sft configs/l20_edu_135m_sft_6k_quality.yaml
```

On a shared GPU box, use the guarded pipeline so it waits for free VRAM instead
of interrupting another process:

```bash
scripts/run_sft_6k_quality_pipeline.sh
```

The default recipe starts from `AliceYin/l20-edu-135m`, uses
`HuggingFaceH4/ultrachat_200k`, masks prompt tokens, and trains only on
assistant response tokens. The recommended comparison is `1k_long` vs
`6k_quality` vs `20k_mixed` under the same sanity eval. Keep `l20-edu-135m` and
`l20-edu-135m-sft` as separate public checkpoints.

## Next Work

The cleanest next pretraining experiment is the controlled baseline:

```bash
python -m l20_pretrain.train configs/l20_wide_140m_baseline.yaml
```

That would test whether the deep-thin architecture is actually better under the
same tokenizer, data slice, context length, optimizer, schedule, and 10B-token
budget.

The most useful product-facing next step is to run the included SFT recipe,
evaluate instruction following, factual QA, short writing, format control, and
base-suite regression, then publish the SFT checkpoint separately from the base
model.

## Sources

- FineWeb-Edu dataset card:
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- SmolLM2-135M model card:
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- EleutherAI lm-evaluation-harness:
  https://github.com/EleutherAI/lm-evaluation-harness
- Hugging Face model cards:
  https://huggingface.co/docs/hub/main/en/model-cards
