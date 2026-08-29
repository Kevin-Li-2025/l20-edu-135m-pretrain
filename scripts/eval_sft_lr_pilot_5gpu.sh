#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
RUN_ROOT="${PROJECT_ROOT}/runs/posttrain/sft-lr-pilot-v1"
OUTPUT_ROOT="${PROJECT_ROOT}/eval_results/posttrain/sft-lr-pilot-v1"
PROMPTS="${PROJECT_ROOT}/configs/posttrain/chat_quality_prompts_v1.jsonl"
mkdir -p "${OUTPUT_ROOT}"

export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false
names=(lr5e5 lr1e4 lr3e4 lr6e4 lr1e3)
pids=()

for gpu in 0 1 2 3 4; do
  name="${names[$gpu]}"
  model="${RUN_ROOT}/${name}/step-000200"
  output="${OUTPUT_ROOT}/${name}"
  if [[ -f "${output}/.complete" ]]; then
    continue
  fi
  for required in config.json model.safetensors tokenizer.json; do
    if [[ ! -s "${model}/${required}" ]]; then
      printf 'missing pilot artifact: %s/%s\n' "${model}" "${required}" >&2
      exit 1
    fi
  done
  mkdir -p "${output}/ifeval"
  (
    if [[ ! -s "${output}/chat_quality.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/eval_chat_quality.py" \
          --model "${model}" \
          --prompts "${PROMPTS}" \
          --output "${output}/chat_quality.json" \
          --device cuda
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}" \
      "${PYTHON_BIN}" -m lm_eval run \
        --model hf \
        --model_args "pretrained=${model},dtype=bfloat16" \
        --tasks ifeval \
        --apply_chat_template \
        --device cuda:0 \
        --batch_size 256 \
        --seed 0,1234,1234,1234 \
        --log_samples \
        --output_path "${output}/ifeval"
    touch "${output}/.complete"
  ) > "${output}/eval.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
