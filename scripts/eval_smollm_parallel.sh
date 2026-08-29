#!/usr/bin/env bash
set -euo pipefail

model_path="${1:?usage: eval_smollm_parallel.sh <model-or-hf-id> <output-dir>}"
output_dir="${2:?usage: eval_smollm_parallel.sh <model-or-hf-id> <output-dir>}"
python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
batch_size="${EVAL_BATCH_SIZE:-auto}"
dtype="${EVAL_DTYPE:-bfloat16}"

mkdir -p "$output_dir"
export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false

task_groups=(
  "lambada_openai"
  "hellaswag"
  "piqa,winogrande"
  "arc_easy"
  "arc_challenge"
)

pids=()
for gpu in 0 1 2 3 4; do
  group_dir="$output_dir/group-$gpu"
  mkdir -p "$group_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m lm_eval \
    --model hf \
    --model_args "pretrained=$model_path,dtype=$dtype" \
    --tasks "${task_groups[$gpu]}" \
    --device cuda:0 \
    --batch_size "$batch_size" \
    --seed 0,1234,1234,1234 \
    --log_samples \
    --output_path "$group_dir" \
    > "$group_dir/eval.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one parallel lm-eval task failed; inspect $output_dir/group-*/eval.log" >&2
  exit "$status"
fi

echo "Parallel six-task evaluation complete: $output_dir"
