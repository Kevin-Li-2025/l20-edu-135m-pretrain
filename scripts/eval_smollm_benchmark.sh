#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,winogrande}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
SEED="${SEED:-20260612}"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/smollm_target_$(date +%Y%m%d_%H%M%S)}"
REQUESTED_CANDIDATE="${CANDIDATE:-}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval is not on PATH. Activate the eval environment first, for example: source .venv-eval/bin/activate" >&2
  exit 2
fi

sanitize_name() {
  echo "$1" | tr '/: ' '___'
}

declare -a MODELS
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
  CANDIDATE="${REQUESTED_CANDIDATE:-${1%%=*}}"
else
  MODELS=(
    "ours-stage2=runs/l20-edu-135m-stage2-math-code-textbook-replay-8k/step-001850"
    "smollm-135m=HuggingFaceTB/SmolLM-135M"
    "smollm2-135m=HuggingFaceTB/SmolLM2-135M"
  )
  if [ -e "runs/l20-edu-135m-stage2-replay-polish-8k/final" ] || [ -e "runs/l20-edu-135m-stage2-replay-polish-8k/step-000300" ]; then
    MODELS=("ours-polish=runs/l20-edu-135m-stage2-replay-polish-8k/final" "${MODELS[@]}")
    CANDIDATE="${REQUESTED_CANDIDATE:-ours-polish}"
  else
    CANDIDATE="${REQUESTED_CANDIDATE:-ours-stage2}"
  fi
fi

mkdir -p "$OUTPUT_ROOT"
declare -a RESULTS

lm_eval validate --tasks "$TASKS"

for entry in "${MODELS[@]}"; do
  if [[ "$entry" != *=* ]]; then
    echo "Expected model entry NAME=MODEL_PATH_OR_HF_ID, got: $entry" >&2
    exit 2
  fi
  name="${entry%%=*}"
  model="${entry#*=}"
  out_dir="$OUTPUT_ROOT/$(sanitize_name "$name")"
  echo "==> Evaluating $name: $model"
  lm_eval run \
    --model hf \
    --model_args "pretrained=${model},dtype=${DTYPE}" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --seed "$SEED" \
    --output_path "$out_dir" \
    --log_samples
  RESULTS+=("--result" "${name}=${out_dir}")
done

"$PYTHON" scripts/summarize_smollm_benchmark.py \
  "${RESULTS[@]}" \
  --candidate "$CANDIDATE" \
  --baseline smollm-135m \
  --baseline smollm2-135m \
  --out-md "$OUTPUT_ROOT/summary.md" \
  --out-json "$OUTPUT_ROOT/summary.json" \
  --out-csv "$OUTPUT_ROOT/summary.csv"
