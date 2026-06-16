---
library_name: transformers
pipeline_tag: text-generation
tags:
- continual-pretraining
- sft
- 135m
- single-gpu
- l20
- data-curation
license: apache-2.0
language:
- en
---

# L20 Edu 135M Stage 4

`l20-edu-135m` is a 134.5M-parameter Llama-style language model trained and improved on a single NVIDIA L20 GPU. The default checkpoint is the selected Stage 4 anti-forgetting release: an 87.5% SFT checkpoint blended with 12.5% Stage 4 base weights, chosen by six-task regression gates.

The key result is token and compute efficiency: the released model uses roughly 13B pretraining/continual-pretraining tokens total (10B initial FineWeb-Edu pretraining + 3B Stage 4 curated continuation), while public 135M SmolLM references use far larger budgets: SmolLM-135M reports 600B pretraining tokens and SmolLM2-135M reports 2T pretraining tokens. That puts this release at about 2.2% of SmolLM-135M token budget and about 0.65% of SmolLM2-135M token budget, trained on one L20 rather than the 64 H100 setup reported by the SmolLM model cards.

## Highlights

- Parameters: 134,515,008
- Hardware: single NVIDIA L20 GPU
- Initial pretraining: 10,001,252,352 FineWeb-Edu tokens
- Stage 4 continuation: 3,000,000,965 curated tokens
- Released pretraining-token total: about 13.0B tokens
- Throughput from the initial 10B run: 38.5k tokens/s mean after warmup
- Context length: 8,192 tokens for Stage 4 continued pretraining
- Data filtering: cross-source MinHash/LSH deduplication, sentence/paragraph deduplication, benchmark decontamination, and LCS overlap removal
- Selection: benchmark regression gates and base/SFT interpolation to preserve general benchmark performance

## Token-Budget Context

| Model | Reported pretraining tokens | Hardware in public card | Relative to this release |
| --- | ---: | --- | ---: |
| L20 Edu 135M Stage 4 | ~13.0B | 1x NVIDIA L20 | 1.00x |
| SmolLM-135M | 600B | 64x H100 | ~46.2x more tokens |
| SmolLM2-135M | 2T | 64x H100 | ~153.8x more tokens |

This comparison is about training-budget context, not a claim of identical data, tokenizer, architecture, or benchmark protocol. The useful takeaway is that the project demonstrates a complete small-model pretraining, curation, evaluation, SFT, and release pipeline under a much smaller single-GPU budget.

## Benchmark Comparison

| Model | Params | Reported tokens | Reported hardware | 6-task Mean | Budget context |
| --- | ---: | ---: | --- | ---: | --- |
| L20 Edu 135M Stage 4 | 134.5M | ~13B | 1x NVIDIA L20 | 0.4150 | 1.00x |
| SmolLM-135M | 135M | 600B | 64x H100 | 0.4767 | ~46.2x tokens |
| SmolLM2-135M | 135M | 2T | 64x H100 | 0.4917 | ~153.8x tokens |
| Qwen2.5-0.5B | 0.49B | not reported in HF card | not reported in HF card | 0.5363 | larger reference |
| OLMo-1B | 1B | 3T | not listed in HF card | 0.5681 | ~230.8x tokens; 1B upper bound |

These are same-protocol self-run numbers on the released checkpoints. The key same-size comparison is SmolLM-135M: this release is 0.0617 mean points behind while using about 2.2% of SmolLM-135M's reported token budget and a single L20 instead of the 64 H100 setup reported in its model card. Qwen2.5-0.5B and OLMo-1B are included as larger reference/upper-bound checkpoints, not same-size baselines.

Detailed task-level scores are included in `eval_results/stage4_release/model_comparison/summary.md`, `summary.csv`, and `summary.json`.

## Selected Six-Task Results

Regression gate: passed. The selected SFT/interpolated release reaches a six-task mean of 0.4150 versus 0.4141 for the Stage 4 base.

| Task | Metric | Score |
| --- | --- | ---: |
| ARC-Challenge | acc_norm,none | 0.2867 |
| ARC-Easy | acc_norm,none | 0.4958 |
| HellaSwag | acc_norm,none | 0.3240 |
| LAMBADA OpenAI | acc,none | 0.2602 |
| PIQA | acc_norm,none | 0.6148 |
| WinoGrande | acc,none | 0.5083 |
| Mean | selected metric average | 0.4150 |

## Stage 4 Base Results

| Task | Metric | Score |
| --- | --- | ---: |
| ARC-Challenge | acc_norm,none | 0.2833 |
| ARC-Easy | acc_norm,none | 0.5046 |
| HellaSwag | acc_norm,none | 0.3243 |
| LAMBADA OpenAI | acc,none | 0.2482 |
| PIQA | acc_norm,none | 0.6181 |
| WinoGrande | acc,none | 0.5059 |

## Data Gate

- Status: `pass`
- Validation tokens: 4,194,398
- Stage 4 indexed documents: 3,312,229
- Indexed sentence/paragraph segments: 34,852,069
- Benchmark-contaminated documents removed: 23
- Benchmark audit: ARC-Challenge, ARC-Easy, HellaSwag, PIQA, LAMBADA OpenAI, and WinoGrande
- Matching: 13-gram candidates plus token LCS overlap >= 0.60 removal
- Deduplication: 64-permutation MinHash with LSH candidate search across sources

## Training And Selection

- Initial model: 10B-token from-scratch FineWeb-Edu run on one L20
- Stage 4 data: high-quality cross-deduplicated educational/code/reasoning mix
- Stage 4 selected base checkpoint: step 2500
- Selected validation loss: 2.9263725876808167
- SFT data: filtered HuggingFaceTB/smol-smoltalk style data for the 135M model
- SFT training rows: 426,842
- Anti-forgetting: model soup/interpolation candidates selected by six-task regression gates

## Reproducibility

Evaluation uses `lm-evaluation-harness` with fixed seed and full zero-shot task datasets for ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI, PIQA, and WinoGrande. Artifacts and summaries are included under `eval_results/` in the model repository.

Generated: 2026-06-16T16:55:19.734757+00:00

## Intended Use

This is a research model for small-model pretraining, data curation, continual pretraining, evaluation, and downstream fine-tuning experiments. Users should independently validate factuality, safety, and task suitability before deployment.
