#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/posttrain/ultrafeedback-dpo-pilot-v1}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_results/posttrain/ultrafeedback-dpo-pilot-v1}"
START_EVAL="${START_EVAL:-${PROJECT_ROOT}/eval_results/posttrain/smol-smoltalk-sft-replay10-v1/step150/broad}"
BASE_EVAL="${BASE_EVAL:-${PROJECT_ROOT}/eval_results/smollm2-continuation-2b-final/candidate}"
PROMPTS="${PROJECT_ROOT}/configs/posttrain/chat_quality_prompts_v1.jsonl"
TASKS="lambada_openai,hellaswag,piqa,winogrande,arc_easy,openbookqa,sciq,arc_challenge,boolq,swag"

export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}"

names=(step025 step050)
models=("${RUN_ROOT}/checkpoint-25" "${RUN_ROOT}/final")
mkdir -p "${EVAL_ROOT}"

pids=()
for index in 0 1; do
  name="${names[$index]}"
  model="${models[$index]}"
  output="${EVAL_ROOT}/${name}"
  chat_gpu="${index}"
  broad_gpu="$((index + 2))"
  for required in config.json model.safetensors tokenizer.json; do
    if [[ ! -s "${model}/${required}" ]]; then
      printf 'missing checkpoint artifact: %s/%s\n' "${model}" "${required}" >&2
      exit 1
    fi
  done
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
  printf 'one or more DPO pilot evaluations failed under %s\n' "${EVAL_ROOT}" >&2
  exit "${status}"
fi

for name in "${names[@]}"; do
  output="${EVAL_ROOT}/${name}"
  for baseline_name in start base; do
    if [[ "${baseline_name}" == start ]]; then
      baseline="${START_EVAL}"
    else
      baseline="${BASE_EVAL}"
    fi
    "${PYTHON_BIN}" -m l20_pretrain.paired_eval \
      --baseline "${baseline}" --candidate "${output}/broad" \
      --confidence 0.95 --out "${output}/paired-vs-${baseline_name}-primary.json"
    "${PYTHON_BIN}" -m l20_pretrain.paired_eval \
      --baseline "${baseline}" --candidate "${output}/broad" \
      --confidence 0.95 --out "${output}/paired-vs-${baseline_name}-development.json" \
      --task openbookqa --task sciq --task boolq --task swag
  done
  touch "${output}/.complete"
done

touch "${EVAL_ROOT}/.complete"
printf 'DPO pilot evaluation complete: %s\n' "${EVAL_ROOT}"
