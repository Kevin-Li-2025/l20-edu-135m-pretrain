#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_results/posttrain/sft-replay-ratio-v1}"
BASELINE_EVAL="${BASELINE_EVAL:-${PROJECT_ROOT}/eval_results/smollm2-continuation-2b-final/candidate}"
PROMPTS="${PROJECT_ROOT}/configs/posttrain/chat_quality_prompts_v1.jsonl"
TASKS="lambada_openai,hellaswag,piqa,winogrande,arc_easy,openbookqa,sciq,arc_challenge,boolq,swag"

export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}"

mkdir -p "${EVAL_ROOT}"
pids=()
for index in 0 1; do
  ratio=$((20 + index * 10))
  name="replay${ratio}"
  model="${PROJECT_ROOT}/runs/posttrain/smol-smoltalk-sft-replay${ratio}-v1/step-000150"
  output="${EVAL_ROOT}/${name}"
  chat_gpu="${index}"
  broad_gpu=$((index + 2))
  mkdir -p "${output}/ifeval" "${output}/broad"

  (
    CUDA_VISIBLE_DEVICES="${chat_gpu}" "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/scripts/eval_chat_quality.py" \
      --model "${model}" --prompts "${PROMPTS}" \
      --output "${output}/chat_quality.json" --device cuda
    CUDA_VISIBLE_DEVICES="${chat_gpu}" "${PYTHON_BIN}" -m lm_eval run \
      --model hf --model_args "pretrained=${model},dtype=bfloat16" \
      --tasks ifeval --apply_chat_template --device cuda:0 --batch_size 256 \
      --seed 0,1234,1234,1234 --log_samples --output_path "${output}/ifeval"
  ) > "${output}/chat-ifeval.log" 2>&1 &
  pids+=("$!")

  (
    CUDA_VISIBLE_DEVICES="${broad_gpu}" "${PYTHON_BIN}" -m lm_eval run \
      --model hf --model_args "pretrained=${model},dtype=bfloat16" \
      --tasks "${TASKS}" --device cuda:0 --batch_size 64 \
      --seed 0,1234,1234,1234 --log_samples --output_path "${output}/broad"
    touch "${output}/broad/.complete"
  ) > "${output}/broad.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  printf 'one or more replay-ratio evaluations failed under %s\n' "${EVAL_ROOT}" >&2
  exit "${status}"
fi

for ratio in 20 30; do
  output="${EVAL_ROOT}/replay${ratio}"
  "${PYTHON_BIN}" -m l20_pretrain.paired_eval \
    --baseline "${BASELINE_EVAL}" --candidate "${output}/broad" \
    --confidence 0.95 --out "${output}/paired-vs-start-primary.json"
  "${PYTHON_BIN}" -m l20_pretrain.paired_eval \
    --baseline "${BASELINE_EVAL}" --candidate "${output}/broad" \
    --confidence 0.95 --out "${output}/paired-vs-start-development.json" \
    --task openbookqa --task sciq --task boolq --task swag
  touch "${output}/.complete"
done

touch "${EVAL_ROOT}/.complete"
printf 'replay ratio evaluation complete: %s\n' "${EVAL_ROOT}"
