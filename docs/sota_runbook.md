# SOTA Runbook

This runbook is intentionally strict. The project should not claim SOTA unless
the gate at the end passes.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scripts/setup_eval_env.sh
python scripts/check_sota_readiness.py
```

The shell scripts default `HF_ENDPOINT` to `https://hf-mirror.com` when the
official Hugging Face endpoint is unreachable. Override `HF_ENDPOINT` if your
network can reach `https://huggingface.co` directly.

The readiness check must show:

- matching tokenizer, data, filters, context, and token budget for the controlled
  configs
- core Python dependencies available
- `lm_eval` importable
- an NVIDIA GPU visible through `nvidia-smi`

## 2. Controlled Training

Train both models, not just the candidate:

```bash
python -m l20_pretrain.train configs/l20_135m_deepthin.yaml
python -m l20_pretrain.train configs/l20_wide_140m_baseline.yaml
```

Copy or symlink the selected final checkpoint to:

```text
runs/l20-edu-135m-deepthin/final
runs/l20-edu-140m-wide-baseline/final
```

The checkpoint selection rule is fixed before evaluation: use the final planned
token checkpoint unless the run crashes, in which case the run is invalid for a
SOTA claim.

The training script updates the `final` symlink automatically whenever it writes
a checkpoint, and keeps only the latest configured checkpoints to avoid filling
the disk.

## 3. Evaluation

```bash
scripts/eval_lm_harness.sh runs/l20-edu-135m-deepthin/final eval_results/l20-edu-135m-deepthin
scripts/eval_lm_harness.sh runs/l20-edu-140m-wide-baseline/final eval_results/l20-edu-140m-wide-baseline
scripts/eval_public_baselines.sh
```

Do not change task names, dtype, or shot settings between models.

## 4. Contamination Check

Create a local training sample file from the exact data stream used in training,
then compare it against lm-eval logs:

```bash
python scripts/sample_training_text.py configs/l20_135m_deepthin.yaml --docs 10000 --out data/train_sample.txt
python scripts/check_contamination.py \
  --train data/train_sample.txt \
  --benchmark eval_results \
  --out eval_results/contamination_report.json
```

The report must pass before any SOTA language is used.

## 5. Comparison and Gate

```bash
python scripts/compare_lm_eval.py \
  --candidate l20-edu-135m-deepthin=eval_results/l20-edu-135m-deepthin \
  --baseline l20-edu-140m-wide-baseline=eval_results/l20-edu-140m-wide-baseline \
  --baseline gpt2-small=eval_results/gpt2-small \
  --baseline opt-125m=eval_results/opt-125m \
  --baseline gpt-neo-125m=eval_results/gpt-neo-125m \
  --baseline cerebras-gpt-111m=eval_results/cerebras-gpt-111m \
  --baseline pythia-160m=eval_results/pythia-160m \
  --baseline smollm-135m=eval_results/smollm-135m \
  --baseline smollm2-135m=eval_results/smollm2-135m

python scripts/sota_gate.py
```

Only `status: pass` in `eval_results/sota_gate.json` authorizes the claim.

Build the release model card after the gate:

```bash
python scripts/build_model_card.py --out MODEL_CARD.md
```

## 6. Claim Language

Allowed after the gate passes:

> Best controlled 100M-150M base model in our same-tokenizer, same-data,
> same-token-budget architecture comparison.

Allowed if public baseline comparisons also pass:

> Competitive with public 100M-200M base models on our fixed lm-eval suite.

Not allowed from this project alone:

> General SOTA language model.

That claim needs larger-scale leaderboards, broader tasks, multiple seeds or
clear uncertainty analysis, public review, and a much larger training run.
