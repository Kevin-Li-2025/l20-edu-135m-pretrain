#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1

VAL_TOKENS="${VAL_TOKENS:-524288}"
TARGET_TOKENS="${TARGET_TOKENS:-8192}"

for DOMAIN in math code textbook; do
  RECIPE="configs/mixtures/eval/l20_eval_${DOMAIN}.yaml"
  OUTPUT_DIR="data/l20_eval_${DOMAIN}_8k"
  python -m l20_pretrain.prepare_mixture_shards \
    --recipe "$RECIPE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens "$TARGET_TOKENS" \
    --val-tokens "$VAL_TOKENS"
done
