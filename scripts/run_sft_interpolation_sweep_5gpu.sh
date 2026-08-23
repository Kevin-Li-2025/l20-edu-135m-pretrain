#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
BASE="${BASE:-${PROJECT_ROOT}/runs/a40-5x-smollm2-continuation-2b-1k-repair-lr3e4/step-002035}"
TUNED="${TUNED:-${PROJECT_ROOT}/runs/posttrain/smol-smoltalk-sft-full-stage1-v1/step-000286}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/posttrain/sft-interpolation-v1}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_results/posttrain/sft-interpolation-v1}"
BASELINE_EVAL="${BASELINE_EVAL:-${PROJECT_ROOT}/eval_results/smollm2-continuation-2b-final/candidate}"
PROMPTS="${PROJECT_ROOT}/configs/posttrain/chat_quality_prompts_v1.jsonl"

export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}"

names=(alpha025 alpha050 alpha075 alpha0875)
alphas=(0.25 0.50 0.75 0.875)
tasks="lambada_openai,hellaswag,piqa,winogrande,arc_easy,openbookqa,sciq,arc_challenge,boolq,swag"
mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}"

for index in 0 1 2 3; do
  model="${RUN_ROOT}/${names[$index]}"
  if [[ ! -s "${model}/model.safetensors" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/interpolate_models.py" \
      --base "${BASE}" \
      --tuned "${TUNED}" \
      --alpha "${alphas[$index]}" \
      --output "${model}"
  fi
done

pids=()
for gpu in 0 1 2 3; do
  name="${names[$gpu]}"
  model="${RUN_ROOT}/${name}"
  output="${EVAL_ROOT}/${name}"
  mkdir -p "${output}/ifeval" "${output}/broad"
  (
    if [[ ! -s "${output}/chat_quality.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
        "${PROJECT_ROOT}/scripts/eval_chat_quality.py" \
        --model "${model}" \
        --prompts "${PROMPTS}" \
        --output "${output}/chat_quality.json" \
        --device cuda
    fi
    if ! find "${output}/ifeval" -name 'results_*.json' -print -quit | grep -q .; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m lm_eval run \
        --model hf \
        --model_args "pretrained=${model},dtype=bfloat16" \
        --tasks ifeval \
        --apply_chat_template \
        --device cuda:0 \
        --batch_size 256 \
        --seed 0,1234,1234,1234 \
        --log_samples \
        --output_path "${output}/ifeval"
    fi
    if ! find "${output}/broad" -name 'results_*.json' -print -quit | grep -q .; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m lm_eval run \
        --model hf \
        --model_args "pretrained=${model},dtype=bfloat16" \
        --tasks "${tasks}" \
        --device cuda:0 \
        --batch_size 64 \
        --seed 0,1234,1234,1234 \
        --log_samples \
        --output_path "${output}/broad"
    fi
    touch "${output}/broad/.complete"
  ) > "${output}/eval.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  printf 'one or more interpolation evaluations failed under %s\n' "${EVAL_ROOT}" >&2
  exit "${status}"
fi

for name in "${names[@]}"; do
  output="${EVAL_ROOT}/${name}"
  "${PYTHON_BIN}" -m l20_pretrain.paired_eval \
    --baseline "${BASELINE_EVAL}" \
    --candidate "${output}/broad" \
    --confidence 0.95 \
    --out "${output}/paired-vs-start-primary.json"
  "${PYTHON_BIN}" -m l20_pretrain.paired_eval \
    --baseline "${BASELINE_EVAL}" \
    --candidate "${output}/broad" \
    --confidence 0.95 \
    --out "${output}/paired-vs-start-development.json" \
    --task openbookqa --task sciq --task boolq --task swag
  touch "${output}/.complete"
done

touch "${EVAL_ROOT}/.complete"
printf 'interpolation sweep complete: %s\n' "${EVAL_ROOT}"
