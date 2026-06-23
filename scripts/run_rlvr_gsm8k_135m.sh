#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
BASE_MODEL="${BASE_MODEL:-runs/l20-edu-135m-stage4-sft-anti-forgetting/interpolated/a0875}"
RUN_DIR="${RUN_DIR:-runs/l20-edu-135m-rlvr-gsm8k-grpo}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-320}"
EVAL_PREFIX="${EVAL_PREFIX:-gsm8k}"
mkdir -p "$LOG_DIR" data/rlvr eval_results/rlvr

if [ "${SKIP_PREPARE:-0}" != "1" ]; then
  "$PYTHON" scripts/prepare_gsm8k_rlvr_data.py \
    --train-output data/rlvr/gsm8k_train.jsonl \
    --eval-output data/rlvr/gsm8k_test.jsonl \
    --summary-output data/rlvr/gsm8k_summary.json
else
  echo "Skipping GSM8K data preparation because SKIP_PREPARE=1"
fi

if [ "${SKIP_BEFORE_EVAL:-0}" != "1" ]; then
  "$PYTHON" scripts/eval_gsm8k_exact.py "$BASE_MODEL" \
    --data data/rlvr/gsm8k_test.jsonl \
    --output "eval_results/rlvr/${EVAL_PREFIX}_before.jsonl" \
    --summary-output "eval_results/rlvr/${EVAL_PREFIX}_before_summary.json" \
    --max-new-tokens "$MAX_COMPLETION_LENGTH" \
    2>&1 | tee "$LOG_DIR/rlvr_${EVAL_PREFIX}_before_eval.log"
else
  echo "Skipping before eval because SKIP_BEFORE_EVAL=1"
fi

accelerate launch scripts/train_rlvr_gsm8k_grpo.py \
  --model "$BASE_MODEL" \
  --train-data data/rlvr/gsm8k_train.jsonl \
  --output-dir "$RUN_DIR" \
  --max-steps "${MAX_STEPS:-250}" \
  --learning-rate "${LR:-2e-6}" \
  --per-device-train-batch-size "${MICRO_BATCH:-2}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-4}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-384}" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  2>&1 | tee "$LOG_DIR/rlvr_${EVAL_PREFIX}_train.log"

"$PYTHON" scripts/eval_gsm8k_exact.py "$RUN_DIR/final" \
  --data data/rlvr/gsm8k_test.jsonl \
  --output "eval_results/rlvr/${EVAL_PREFIX}_after.jsonl" \
  --summary-output "eval_results/rlvr/${EVAL_PREFIX}_after_summary.json" \
  --max-new-tokens "$MAX_COMPLETION_LENGTH" \
  2>&1 | tee "$LOG_DIR/rlvr_${EVAL_PREFIX}_after_eval.log"
