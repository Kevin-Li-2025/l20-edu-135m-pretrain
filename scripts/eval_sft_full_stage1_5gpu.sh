#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
MODEL="${MODEL:-${PROJECT_ROOT}/runs/posttrain/smol-smoltalk-sft-full-stage1-v1/step-000286}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_results/posttrain/smol-smoltalk-sft-full-stage1-v1}"
BASELINE_EVAL="${BASELINE_EVAL:-${PROJECT_ROOT}/eval_results/smollm2-continuation-2b-final/candidate}"
PROMPTS="${PROJECT_ROOT}/configs/posttrain/chat_quality_prompts_v1.jsonl"

export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}"

if [[ -f "${OUTPUT_ROOT}/.complete" ]]; then
  printf 'evaluation already complete: %s\n' "${OUTPUT_ROOT}"
  exit 0
fi
for required in config.json model.safetensors tokenizer.json; do
  if [[ ! -s "${MODEL}/${required}" ]]; then
    printf 'missing checkpoint artifact: %s/%s\n' "${MODEL}" "${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${BASELINE_EVAL}/.complete" ]]; then
  printf 'incomplete frozen baseline: %s\n' "${BASELINE_EVAL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/chat-ifeval/ifeval" "${OUTPUT_ROOT}/broad"
pids=()

(
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/eval_chat_quality.py" \
    --model "${MODEL}" \
    --prompts "${PROMPTS}" \
    --output "${OUTPUT_ROOT}/chat-ifeval/chat_quality.json" \
    --device cuda
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m lm_eval run \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks ifeval \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size 256 \
    --seed 0,1234,1234,1234 \
    --log_samples \
    --output_path "${OUTPUT_ROOT}/chat-ifeval/ifeval"
) > "${OUTPUT_ROOT}/chat-ifeval/eval.log" 2>&1 &
pids+=("$!")

task_groups=(
  "lambada_openai"
  "hellaswag"
  "piqa,winogrande"
  "arc_easy,openbookqa,sciq"
  "arc_challenge,boolq,swag"
)

for gpu in 1 2 3; do
  group=$((gpu - 1))
  group_dir="${OUTPUT_ROOT}/broad/group-${group}"
  mkdir -p "${group_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m lm_eval run \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks "${task_groups[$group]}" \
    --device cuda:0 \
    --batch_size 64 \
    --seed 0,1234,1234,1234 \
    --log_samples \
    --output_path "${group_dir}" \
    > "${group_dir}/eval.log" 2>&1 &
  pids+=("$!")
done

(
  for group in 3 4; do
    group_dir="${OUTPUT_ROOT}/broad/group-${group}"
    mkdir -p "${group_dir}"
    CUDA_VISIBLE_DEVICES=4 "${PYTHON_BIN}" -m lm_eval run \
      --model hf \
      --model_args "pretrained=${MODEL},dtype=bfloat16" \
      --tasks "${task_groups[$group]}" \
      --device cuda:0 \
      --batch_size 64 \
      --seed 0,1234,1234,1234 \
      --log_samples \
      --output_path "${group_dir}" \
      > "${group_dir}/eval.log" 2>&1
  done
) &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  printf 'one or more frozen evaluation jobs failed under %s\n' "${OUTPUT_ROOT}" >&2
  exit "${status}"
fi

touch "${OUTPUT_ROOT}/broad/.complete"
"${PYTHON_BIN}" -m l20_pretrain.paired_eval \
  --baseline "${BASELINE_EVAL}" \
  --candidate "${OUTPUT_ROOT}/broad" \
  --confidence 0.95 \
  --out "${OUTPUT_ROOT}/paired-vs-start-primary.json"
"${PYTHON_BIN}" -m l20_pretrain.paired_eval \
  --baseline "${BASELINE_EVAL}" \
  --candidate "${OUTPUT_ROOT}/broad" \
  --confidence 0.95 \
  --out "${OUTPUT_ROOT}/paired-vs-start-development.json" \
  --task openbookqa --task sciq --task boolq --task swag

touch "${OUTPUT_ROOT}/.complete"
printf 'stage-1 frozen evaluation complete: %s\n' "${OUTPUT_ROOT}"
