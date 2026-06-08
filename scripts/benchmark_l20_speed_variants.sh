#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/l20_edu_135m_hq_longctx_8k.yaml}"
VARIANTS="${VARIANTS:-base:2,base:3,base:4,ckpt:5,ckpt:6,compile:2,liger:2,liger:3,liger+compile:2}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
MEASURE_STEPS="${MEASURE_STEPS:-5}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
OUTPUT_JSONL="${OUTPUT_JSONL:-docs/l20_speed_benchmark.jsonl}"

mkdir -p "$(dirname "$OUTPUT_JSONL")"

python scripts/benchmark_train_variants.py "$CONFIG" \
  --variants "$VARIANTS" \
  --warmup-steps "$WARMUP_STEPS" \
  --measure-steps "$MEASURE_STEPS" \
  --grad-accumulation-steps "$GRAD_ACCUMULATION_STEPS" \
  --output-jsonl "$OUTPUT_JSONL"
