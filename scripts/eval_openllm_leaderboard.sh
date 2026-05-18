#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: scripts/eval_openllm_leaderboard.sh <model-or-checkpoint> [output-dir]}"
MODEL_NAME="$(basename "$MODEL_PATH")"
OUT_DIR="${2:-eval_results/${MODEL_NAME}-leaderboard}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=${DTYPE}" \
  --tasks leaderboard \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR" \
  --log_samples
