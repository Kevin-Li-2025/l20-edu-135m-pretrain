#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1

TARGET_TOKENS="${TARGET_TOKENS:-700000000}"
VAL_TOKENS="${VAL_TOKENS:-4194304}"
OUTPUT_DIR="${OUTPUT_DIR:-data/l20_edu_hq_8k}"

set +e
python -m l20_pretrain.prepare_shards \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer AliceYin/l20-edu-135m \
  --dataset HuggingFaceFW/fineweb-edu \
  --config-name sample-10BT \
  --split train \
  --text-column text \
  --target-tokens "$TARGET_TOKENS" \
  --val-tokens "$VAL_TOKENS" \
  --block-size 8192 \
  --min-chars 700 \
  --max-chars 45000 \
  --min-score 3.0 \
  --min-int-score 3 \
  --report-interval 500
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ] && [ -f "$OUTPUT_DIR/metadata.json" ]; then
  echo "prepare_python_exit_code=$STATUS after metadata was written; treating shards as complete"
  exit 0
fi

exit "$STATUS"
