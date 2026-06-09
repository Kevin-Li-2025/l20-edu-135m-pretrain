#!/usr/bin/env bash
set -euo pipefail

# 默认加载 Stage 2 跑到 1850 步的最终模型
MODEL_PATH="${1:-/home/hhai/l20-pretrain/out/l20_stage2_math_code_textbook_replay_8k/step-001850}"
MODEL_NAME="$(basename "$MODEL_PATH")"
OUT_DIR="${2:-eval_results/${MODEL_NAME}_sota}"

# 涵盖四大底盘指标 + Stage 2 的数学杀手锏 (GSM8K, MathQA)
TASKS="hellaswag,piqa,arc_easy,arc_challenge,mmlu,gsm8k"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "🚀 正在启动针对 $MODEL_NAME 的 SOTA 大考..."
echo "📊 测试任务列表: $TASKS"
echo "💾 结果将保存在: $OUT_DIR"

# 运行 lm-eval
lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=${DTYPE}" \
  --tasks "$TASKS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR" \
  --log_samples

echo "✅ 跑分完成！"
