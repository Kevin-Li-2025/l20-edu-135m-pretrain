#!/usr/bin/env bash
set -euo pipefail

cd /home/hhai/l20-pretrain
export PATH="$PWD/.venv-continue/bin:$PWD/.venv-eval/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs data/sft eval_results/behavior_repair_current

python scripts/prepare_behavior_sft_data.py \
  --output data/sft/behavior_repair_train.jsonl \
  --eval-output data/sft/behavior_repair_eval.jsonl \
  --repeat 24 \
  --eval-size 64 \
  --seed 20260616 \
  2>&1 | tee logs/behavior_repair_prepare.log

python -m l20_pretrain.train_sft configs/l20_edu_135m_sft_behavior_repair_current.yaml \
  2>&1 | tee logs/behavior_repair_train.log

python scripts/eval_sft_sanity.py runs/l20-edu-135m-sft-behavior-repair-current/final \
  --output eval_results/behavior_repair_current/sanity.jsonl \
  --markdown-output eval_results/behavior_repair_current/sanity.md \
  --max-new-tokens 80 \
  2>&1 | tee logs/behavior_repair_sanity.log

CANDIDATE=behavior-repair-current OUTPUT_ROOT=eval_results/behavior_repair_current/six_tasks \
  bash scripts/eval_smollm_benchmark.sh \
    "behavior-repair-current=runs/l20-edu-135m-sft-behavior-repair-current/final" \
    2>&1 | tee logs/behavior_repair_six_tasks.log
