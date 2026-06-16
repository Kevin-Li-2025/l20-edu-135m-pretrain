#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

while [ ! -s eval_results/rlvr/gsm8k_after_summary.json ]; do
  sleep 60
done

while pgrep -af "scripts/eval_gsm8k_exact.py|scripts/train_rlvr_gsm8k_grpo.py" | grep -v grep >/dev/null; do
  sleep 60
done

export PATH="$PWD/.venv-eval/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHON="${PYTHON:-$PWD/.venv-eval/bin/python}"
export RUN_DIR="${RUN_DIR:-runs/l20-edu-135m-rlvr-gsm8k-grpo-c320}"
export EVAL_PREFIX="${EVAL_PREFIX:-gsm8k_c320}"
export MAX_STEPS="${MAX_STEPS:-350}"
export LR="${LR:-1e-6}"
export MICRO_BATCH="${MICRO_BATCH:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-320}"
export SKIP_BEFORE_EVAL="${SKIP_BEFORE_EVAL:-1}"

bash scripts/run_rlvr_gsm8k_135m.sh
