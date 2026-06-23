#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/l20_edu_135m_gsm8k_cot_warmup.yaml}"
WARMUP_RUN_DIR="${WARMUP_RUN_DIR:-runs/l20-edu-135m-gsm8k-cot-warmup}"
RLVR_RUN_DIR="${RLVR_RUN_DIR:-runs/l20-edu-135m-gsm8k-cot-warmup-rlvr-c320}"
EVAL_PREFIX="${EVAL_PREFIX:-gsm8k_cot_warmup_c320}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" data/sft eval_results/rlvr

"$PYTHON" scripts/prepare_gsm8k_cot_sft_data.py \
  --train-output data/sft/gsm8k_cot_train.jsonl \
  --eval-output data/sft/gsm8k_cot_eval.jsonl \
  --summary-output data/sft/gsm8k_cot_summary.json \
  2>&1 | tee "$LOG_DIR/gsm8k_cot_sft_prepare.log"

"$PYTHON" -m l20_pretrain.train_sft "$CONFIG" \
  2>&1 | tee "$LOG_DIR/gsm8k_cot_sft_train.log"

"$PYTHON" scripts/eval_gsm8k_exact.py "$WARMUP_RUN_DIR/final" \
  --data data/rlvr/gsm8k_test.jsonl \
  --output "eval_results/rlvr/${EVAL_PREFIX}_warmup.jsonl" \
  --summary-output "eval_results/rlvr/${EVAL_PREFIX}_warmup_summary.json" \
  --max-new-tokens "${MAX_COMPLETION_LENGTH:-320}" \
  2>&1 | tee "$LOG_DIR/rlvr_${EVAL_PREFIX}_warmup_eval.log"

BASE_MODEL="$WARMUP_RUN_DIR/final" \
RUN_DIR="$RLVR_RUN_DIR" \
EVAL_PREFIX="$EVAL_PREFIX" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-320}" \
MAX_STEPS="${MAX_STEPS:-350}" \
LR="${LR:-8e-7}" \
MICRO_BATCH="${MICRO_BATCH:-2}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-4}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}" \
SKIP_PREPARE="${SKIP_PREPARE:-0}" \
SKIP_BEFORE_EVAL="${SKIP_BEFORE_EVAL:-1}" \
bash scripts/run_rlvr_gsm8k_135m.sh
