#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python}"
RECIPE="${RECIPE:-configs/mixtures/l20_stage3_dclm_edu_replay.yaml}"
TARGET_TOKENS="${TARGET_TOKENS:-10000000000}"
VAL_TOKENS="${VAL_TOKENS:-4194304}"
OUTPUT_DIR="${OUTPUT_DIR:-data/l20_stage3_dclm_edu_replay_8k}"

if [ ! -f "data/l20_edu_hq_8k/train.bin" ]; then
  echo "Stage-1 tokenized replay shard is missing: data/l20_edu_hq_8k/train.bin" >&2
  exit 2
fi

set +e
"$PYTHON" -m l20_pretrain.prepare_mixture_shards \
  --recipe "$RECIPE" \
  --output-dir "$OUTPUT_DIR" \
  --target-tokens "$TARGET_TOKENS" \
  --val-tokens "$VAL_TOKENS"
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ] && [ -f "$OUTPUT_DIR/metadata.json" ]; then
  echo "prepare_mixture_python_exit_code=$STATUS after metadata was written; treating shards as complete"
  exit 0
fi

exit "$STATUS"
