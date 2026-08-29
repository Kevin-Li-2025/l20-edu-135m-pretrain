# L20-Edu-135M

An auditable single-GPU study of data-efficient 135M language-model training.
The project trains a Llama-style decoder model from scratch, continues it on a
strictly filtered Stage 4 mixture, evaluates public baselines under the same
`lm-eval` harness, and records a small-model RLVR negative result. A separate
A40 extension adds multi-GPU continual-pretraining and capacity-aware
post-training infrastructure without changing the selected public checkpoint.

- Model: [AliceYin/l20-edu-135m](https://huggingface.co/AliceYin/l20-edu-135m)
- Paper draft: [paper/l20_edu_135m_arxiv.pdf](paper/l20_edu_135m_arxiv.pdf)
- Technical report: [docs/project_report/TECHNICAL_REPORT.md](docs/project_report/TECHNICAL_REPORT.md)
- Next ablations: [docs/project_report/ablation_plan.json](docs/project_report/ablation_plan.json)
- A40 runbook: [A40_RUNBOOK.md](A40_RUNBOOK.md)
- A40 efficiency study: [docs/A40_MAX_EFFICIENCY_RESEARCH.md](docs/A40_MAX_EFFICIENCY_RESEARCH.md)
- Curated result files: [results/](results/)

## Why This Repo Exists

Most public small-language-model releases hide the parts that matter for
reproducibility: exact token budgets, filtering gates, continuation decisions,
failed post-training experiments, and compute constraints. This repository keeps
those pieces visible.

The central result is not a state-of-the-art claim. It is a controlled,
single-NVIDIA-L20 training record showing how far a 135M model can be pushed
with about 13B total tokens, careful data filtering, conservative SFT
interpolation, and honest evaluation against larger-budget public baselines.

## Headline Result

| Model | Params | Public/prepared tokens | Hardware | 6-task mean |
| --- | ---: | ---: | --- | ---: |
| L20-Edu-135M | 134.5M | ~13B | 1x L20 | 0.4150 |
| SmolLM-135M | 135M | 600B | 64x H100 | 0.4767 |
| SmolLM2-135M | 135M | 2T | 64x H100 | 0.4917 |
| Qwen2.5-0.5B | 494M | public | public | 0.5363 |
| OLMo-1B | 1B | public | public | 0.5681 |

The L20-Edu checkpoint reaches about 87.1% of the self-run SmolLM-135M six-task
mean with about 2.17% of its reported token budget. The same comparison against
SmolLM2-135M is about 84.4% of the mean with about 0.65% of the token budget.

The six-task suite is ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI, PIQA,
and WinoGrande. Baseline numbers are self-run with the same harness protocol
where possible, not copied from leaderboards.

## Final Selected Checkpoint

The selected public checkpoint is the Stage 4 SFT interpolation candidate
`stage4-sft-a0875`: 87.5% anti-forgetting SFT interpolation and 12.5% Stage 4
base interpolation. This was selected because it gave the best aggregate score
without clear regression on the base benchmark suite.

| Task | Metric | Score |
| --- | --- | ---: |
| ARC-Challenge | acc_norm | 0.2867 |
| ARC-Easy | acc_norm | 0.4958 |
| HellaSwag | acc_norm | 0.3240 |
| LAMBADA OpenAI | acc | 0.2602 |
| PIQA | acc_norm | 0.6148 |
| WinoGrande | acc | 0.5083 |
| Mean | - | 0.4150 |

## Data And Contamination Controls

Stage 4 added 3,000,000,965 curated continuation tokens after the initial 10B
FineWeb-Edu run. The filtering gate included:

- cross-source 64-permutation MinHash plus LSH near-deduplication;
- sentence/template and repeated-paragraph filtering;
- benchmark 13-gram contamination screening;
- LCS overlap removal at ratio `>= 0.60`;
- per-source epoch caps to avoid overtraining narrow sources;
- restricted MixtureVita usage focused on structured tutorial, FAQ, and
  reasoning-like content because the source card does not claim complete
  cross-source deduplication.

The recorded gate indexed 3,312,229 documents, checked 34,852,069 segments,
created 52,995,632 LSH bands, and removed 23 benchmark-overlap candidates.

## Post-Training Findings

SFT helped slightly only when mixed back with the base checkpoint. Full SFT did
not dominate the base model on the six-task suite, so the release keeps the
interpolation result rather than overstating instruction-tuning gains.

RLVR on GSM8K was negative at this scale. The best recorded base score was
24/1319 exact matches, while RLVR variants decreased exact-match accuracy. The
repo includes this result because it is useful evidence for the question of
whether verifiable-reward reasoning emerges in a 135M model.

## Repository Map

```text
configs/                 Training, continuation, SFT, and benchmark configs
docs/                    Reports, training notes, curves, and design notes
paper/                   arXiv-style paper draft and bibliography
results/                 Curated benchmark, Stage 4, and RLVR result summaries
scripts/                 Data prep, evaluation, reporting, and RLVR utilities
src/l20_pretrain/        Model, data pipeline, training, SFT, and reward code
tests/                   Unit tests for parsers, data code, and RLVR rewards
A40_RUNBOOK.md           Five/six-A40 DDP deployment and topology checks
```

Large artifacts are intentionally not committed: checkpoints, raw datasets,
raw `lm-eval` sample dumps, run directories, logs, Hugging Face tokens, and
machine-local environment directories stay outside Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install the PyTorch build that matches your CUDA runtime before training on a
GPU host.

## Reproduce Key Checks

Run repository hygiene checks:

```bash
python scripts/check_repo_hygiene.py
```

Run unit tests:

```bash
python -m pytest -q
```

Run the six-task harness for a local checkpoint:

```bash
scripts/eval_lm_harness.sh runs/l20-edu-135m-stage4-selected
```

Run public baselines and compare:

```bash
scripts/eval_public_baselines.sh
python scripts/compare_lm_eval.py \
  --candidate l20-edu-135m=eval_results/l20-edu-135m \
  --baseline smollm-135m=eval_results/smollm-135m \
  --baseline smollm2-135m=eval_results/smollm2-135m
```

Run the GSM8K exact-match summarizer:

```bash
python scripts/eval_gsm8k_exact.py --help
python scripts/summarize_rlvr_gsm8k_results.py --help
```

For the multi-GPU continuation and post-training extension, start with
[`A40_RUNBOOK.md`](A40_RUNBOOK.md). Its host-specific NCCL settings must be
remeasured before reuse on a different cluster. Candidate checkpoints remain
unpromoted until the paired benchmark and per-task regression gates pass.

## Release Discipline

The project is deliberately conservative:

- claims are tied to committed result summaries and report tables;
- raw samples and benchmark dumps are excluded from Git;
- negative RLVR and SFT ablation outcomes are preserved;
- token-budget comparisons are separated from quality claims;
- no no-contamination claim is made beyond the documented filters.

## Next Research Step

The next improvement target is controlled evidence rather than another
unstructured continuation run. The committed plan tracks five experiments:

- relaxed-filter control for the Stage 4 cleaning gate;
- edu/math/code mixture-ratio ablation;
- 2K/4K/8K sequence-length curriculum comparison;
- SFT quality and checkpoint-interpolation study;
- RLVR scale-threshold study for 135M versus larger future bases.

Validate the plan with:

```bash
python scripts/check_ablation_plan.py
```

Prepare skill-targeted data with the same cleaning and contamination gate:

```bash
python scripts/prepare_skill_targeted_corpus.py data/raw_skill_mix \
  --out data/skill_targeted/clean.jsonl \
  --guard-index data/skill_targeted/cross_source_guard.sqlite \
  --contamination-path data/benchmark_contamination/eval_5tasks.jsonl
```

Reweight the next curriculum stage from benchmark gaps:

```bash
python scripts/eval_and_reweight_mixture.py \
  --scores results/stage4/final_model.json \
  --out results/ablations/next_mixture_weights.json
```

Audit an L20 training config for MFU/tokens/sec risks:

```bash
python scripts/audit_l20_mfu_config.py configs/l20_edu_135m_benchmark_4k.yaml
```

For citation metadata, use [CITATION.cff](CITATION.cff). For a reproducibility
manifest, see [docs/reproducibility.md](docs/reproducibility.md).
