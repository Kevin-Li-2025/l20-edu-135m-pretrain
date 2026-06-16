#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: scripts/run_chat_model_eval.sh <model-or-checkpoint> [output-root]}"
MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_ROOT="${2:-eval_results/chat/${MODEL_NAME}}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
CHAT_LM_EVAL_TASKS="${CHAT_LM_EVAL_TASKS:-ifeval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "model": "$MODEL_PATH",
    "output_root": "$OUTPUT_ROOT",
    "chat_lm_eval_tasks": "$CHAT_LM_EVAL_TASKS",
    "max_new_tokens": int("$MAX_NEW_TOKENS"),
    "created_at": datetime.now(timezone.utc).isoformat(),
}
Path("$OUTPUT_ROOT").mkdir(parents=True, exist_ok=True)
Path("$OUTPUT_ROOT/metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
PY

"$PYTHON" scripts/eval_sft_sanity.py "$MODEL_PATH" \
  --output "$OUTPUT_ROOT/sft_sanity.jsonl" \
  --markdown-output "$OUTPUT_ROOT/sft_sanity.md" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --device "$DEVICE"

if [ ! -s data/rlvr/gsm8k_test.jsonl ]; then
  "$PYTHON" scripts/prepare_gsm8k_rlvr_data.py \
    --train-output data/rlvr/gsm8k_train.jsonl \
    --eval-output data/rlvr/gsm8k_test.jsonl \
    --summary-output data/rlvr/gsm8k_summary.json
fi

"$PYTHON" scripts/eval_gsm8k_exact.py "$MODEL_PATH" \
  --data data/rlvr/gsm8k_test.jsonl \
  --output "$OUTPUT_ROOT/gsm8k_exact.jsonl" \
  --summary-output "$OUTPUT_ROOT/gsm8k_exact_summary.json" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE"

if command -v lm_eval >/dev/null 2>&1; then
  lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=${DTYPE}" \
    --tasks "$CHAT_LM_EVAL_TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_ROOT/lm_eval" \
    --log_samples
else
  echo "lm_eval not found; skipped lm-eval chat tasks: $CHAT_LM_EVAL_TASKS" | tee "$OUTPUT_ROOT/lm_eval_skipped.txt"
fi

"$PYTHON" scripts/summarize_rlvr_gsm8k_results.py \
  --result "gsm8k_exact=$OUTPUT_ROOT/gsm8k_exact.jsonl" \
  --out-json "$OUTPUT_ROOT/gsm8k_diagnostics.json" \
  --out-md "$OUTPUT_ROOT/gsm8k_diagnostics.md"
