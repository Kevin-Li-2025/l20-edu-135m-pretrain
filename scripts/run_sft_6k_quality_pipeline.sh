#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/l20_edu_135m_sft_6k_quality.yaml}"
RUN_NAME="${RUN_NAME:-l20-edu-135m-sft-6k-quality}"
MAX_GPU_USED_MB="${MAX_GPU_USED_MB:-8000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

mkdir -p logs data/sft eval_results/sft_sanity_6k

if [[ ! -s data/sft/ultrachat_6k_quality.jsonl || ! -s data/sft/ultrachat_eval_512.jsonl ]]; then
  python scripts/prepare_sft_data.py \
    --strategy quality \
    --target-size 6000 \
    --eval-size 512 \
    --source-limit 250000 \
    --output data/sft/ultrachat_6k_quality.jsonl \
    --eval-output data/sft/ultrachat_eval_512.jsonl \
    --summary-output data/sft/ultrachat_6k_quality_summary.json
fi

while true; do
  used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "${used_mb}" =~ ^[0-9]+$ && "${used_mb}" -le "${MAX_GPU_USED_MB}" ]]; then
    break
  fi
  printf '{"event":"wait_gpu","memory_used_mb":%s,"max_gpu_used_mb":%s}\n' "${used_mb:-0}" "${MAX_GPU_USED_MB}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

python -m l20_pretrain.train_sft "${CONFIG}"

python scripts/eval_sft_sanity.py "runs/${RUN_NAME}/final" \
  --output "eval_results/sft_sanity_6k/results.jsonl" \
  --markdown-output "eval_results/sft_sanity_6k/report.md"
