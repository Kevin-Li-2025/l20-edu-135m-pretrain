#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PARQUET_RANGE_CHUNK_BYTES="${PARQUET_RANGE_CHUNK_BYTES:-16777216}"
export PARQUET_RANGE_WORKERS="${PARQUET_RANGE_WORKERS:-12}"
export PARQUET_CHUNK_MAX_SECONDS="${PARQUET_CHUNK_MAX_SECONDS:-900}"
export PARQUET_MIN_BYTES_PER_SEC="${PARQUET_MIN_BYTES_PER_SEC:-65536}"
export PARQUET_LOW_SPEED_SECONDS="${PARQUET_LOW_SPEED_SECONDS:-30}"
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

PYTHON="${PYTHON:-.venv-continue/bin/python}"
RECIPE="${RECIPE:-configs/mixtures/l20_stage4_hq_crossdedup.yaml}"
TARGET_TOKENS="${TARGET_TOKENS:-3000000000}"
VAL_TOKENS="${VAL_TOKENS:-4194304}"
OUTPUT_DIR="${OUTPUT_DIR:-data/l20_stage4_hq_crossdedup_8k}"
BUILD_MARKER="$OUTPUT_DIR/.build_in_progress"

if [ -e "$OUTPUT_DIR/metadata.json" ]; then
  echo "Stage 4 data already has metadata: $OUTPUT_DIR/metadata.json"
  exit 0
fi

RESUME_ARGS=()
if [ -s "$OUTPUT_DIR/resume_state.pkl" ]; then
  RESUME_ARGS=(--resume)
  rm -f "$OUTPUT_DIR/.build_failed"
elif [ -d "$OUTPUT_DIR" ] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  STALE_DIR="${OUTPUT_DIR}.stale.$(date -u +%Y%m%dT%H%M%SZ)"
  echo "Moving inconsistent partial build to $STALE_DIR" >&2
  mv "$OUTPUT_DIR" "$STALE_DIR"
fi
mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$(date -Is)" > "$BUILD_MARKER"

cleanup_marker() {
  status=$?
  trap - EXIT
  if [ "$status" -eq 0 ]; then
    rm -f "$BUILD_MARKER"
  else
    printf '%s\n' "$status" > "$OUTPUT_DIR/.build_failed"
  fi
  exit "$status"
}
trap cleanup_marker EXIT

"$PYTHON" scripts/build_benchmark_contamination_index.py \
  --out data/benchmark_contamination/eval_5tasks.jsonl

while true; do
  set +e
  "$PYTHON" -m l20_pretrain.prepare_mixture_shards \
    --recipe "$RECIPE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens "$TARGET_TOKENS" \
    --val-tokens "$VAL_TOKENS" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL:-10000}" \
    --max-rss-gb "${MAX_RSS_GB:-11.5}" \
    "${RESUME_ARGS[@]}"
  status=$?
  set -e
  if [ "$status" -ne 75 ]; then
    exit "$status"
  fi
  echo "Data builder requested a controlled memory recycle; resuming." >&2
  RESUME_ARGS=(--resume)
  sleep 5
done
