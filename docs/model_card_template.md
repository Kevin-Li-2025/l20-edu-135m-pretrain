# Model Card Template

## Model

- Name:
- Type: base causal language model
- Parameter count:
- Architecture:
- Tokenizer:
- Context length:
- License:

## Training

- Training code commit:
- Config file:
- Data:
- Data filters:
- Token budget:
- Optimizer:
- LR schedule:
- Precision:
- Hardware:
- Training time:
- Checkpoint selection rule:

## Evaluation

- Harness:
- Harness commit or version:
- Dtype:
- Batch size:
- Tasks:
- Raw result path:
- Sample log path:

| Task | Shots | Metric | Score |
| --- | ---: | --- | ---: |
| LAMBADA OpenAI | 0 | accuracy / perplexity | |
| HellaSwag | 0 | normalized accuracy | |
| PIQA | 0 | accuracy | |
| ARC-Easy | 0 | accuracy | |
| ARC-Challenge | 0 | accuracy | |
| WinoGrande | 0 | accuracy | |

## Comparison

- Controlled baseline:
- Public baselines:
- Same-token-budget comparison:
- Public-benchmark comparison:

## Contamination Statement

- Benchmarks checked:
- Method:
- Known overlaps:
- Mitigation:

## Limitations

- Not instruction tuned.
- Not safety aligned.
- English-heavy pretraining data.
- Benchmark results should not be interpreted as general assistant quality.

## Intended Use

- Research on small base-model pretraining.
- Architecture and data ablations.
- Local experimentation after downstream fine-tuning.
