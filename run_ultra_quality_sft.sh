#!/usr/bin/env bash
set -euo pipefail

cd /home/hhai/l20-pretrain
export PATH="$PWD/.venv-continue/bin:$PWD/.venv-eval/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs data/sft eval_results/ultra_quality_sft

if [ ! -s data/sft/ultra_quality_sft_30k.jsonl ]; then
  python scripts/prepare_ultra_quality_sft.py \
    --inputs data/sft/stage4_smol_smoltalk_hq.jsonl data/sft/ultrachat_6k_quality.jsonl \
    --output data/sft/ultra_quality_sft_30k.jsonl \
    --eval-output data/sft/ultra_quality_sft_eval_1k.jsonl \
    --summary-output data/sft/ultra_quality_sft_summary.json \
    --target-size 30000 \
    --eval-size 1024 \
    2>&1 | tee logs/ultra_quality_sft_prepare.log
fi

python -m l20_pretrain.train_sft configs/l20_edu_135m_ultra_quality_sft.yaml \
  2>&1 | tee logs/ultra_quality_sft_train.log

python scripts/eval_sft_sanity.py runs/l20-edu-135m-ultra-quality-sft/final \
  --output eval_results/ultra_quality_sft/sanity.jsonl \
  --markdown-output eval_results/ultra_quality_sft/sanity.md \
  2>&1 | tee logs/ultra_quality_sft_sanity.log

if command -v lm_eval >/dev/null 2>&1; then
  CANDIDATE=ultra-quality-sft OUTPUT_ROOT=eval_results/ultra_quality_sft/six_tasks \
    bash scripts/eval_smollm_benchmark.sh \
      "ultra-quality-sft=runs/l20-edu-135m-ultra-quality-sft/final" \
      2>&1 | tee logs/ultra_quality_sft_six_tasks.log
fi
