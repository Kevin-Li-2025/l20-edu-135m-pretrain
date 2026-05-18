#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-configs/sota_eval.yaml}"
TRAIN="${TRAIN:-0}"
EVAL="${EVAL:-1}"
RUN_PUBLIC_BASELINES="${RUN_PUBLIC_BASELINES:-1}"
CONTAMINATION_TRAIN_SAMPLE="${CONTAMINATION_TRAIN_SAMPLE:-data/train_sample.txt}"
CONTAMINATION_DOCS="${CONTAMINATION_DOCS:-10000}"
CONTAMINATION_BENCHMARK_DIR="${CONTAMINATION_BENCHMARK_DIR:-eval_results}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python scripts/check_sota_readiness.py --manifest "$MANIFEST"
python scripts/model_report.py configs/l20_135m_deepthin.yaml configs/l20_wide_140m_baseline.yaml

if [[ "$TRAIN" == "1" ]]; then
  python -m l20_pretrain.train configs/l20_135m_deepthin.yaml
  python -m l20_pretrain.train configs/l20_wide_140m_baseline.yaml
fi

if [[ "$EVAL" == "1" ]]; then
  scripts/eval_lm_harness.sh runs/l20-edu-135m-deepthin/final eval_results/l20-edu-135m-deepthin
  scripts/eval_lm_harness.sh runs/l20-edu-140m-wide-baseline/final eval_results/l20-edu-140m-wide-baseline
  if [[ "$RUN_PUBLIC_BASELINES" == "1" ]]; then
    scripts/eval_public_baselines.sh
  fi
fi

if [[ ! -f "$CONTAMINATION_TRAIN_SAMPLE" ]]; then
  python scripts/sample_training_text.py \
    configs/l20_135m_deepthin.yaml \
    --docs "$CONTAMINATION_DOCS" \
    --out "$CONTAMINATION_TRAIN_SAMPLE"
fi

python scripts/check_contamination.py \
  --train "$CONTAMINATION_TRAIN_SAMPLE" \
  --benchmark "$CONTAMINATION_BENCHMARK_DIR" \
  --out eval_results/contamination_report.json

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

python scripts/sota_gate.py --manifest "$MANIFEST"
