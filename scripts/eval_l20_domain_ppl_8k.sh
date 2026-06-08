#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CHECKPOINT="${1:-runs/l20-edu-135m-hq-longctx-8k/final}"
DTYPE="${DTYPE:-bfloat16}"

declare -a CONFIGS=(
  "general:configs/eval/l20_eval_general_edu_8k.yaml"
  "math:configs/eval/l20_eval_math_8k.yaml"
  "code:configs/eval/l20_eval_code_8k.yaml"
  "textbook:configs/eval/l20_eval_textbook_8k.yaml"
)

for ITEM in "${CONFIGS[@]}"; do
  DOMAIN="${ITEM%%:*}"
  CONFIG="${ITEM#*:}"
  echo "domain=$DOMAIN config=$CONFIG checkpoint=$CHECKPOINT"
  python -m l20_pretrain.eval_ppl "$CHECKPOINT" "$CONFIG" --dtype "$DTYPE"
done
