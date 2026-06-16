#!/usr/bin/env bash
set -euo pipefail

cd /home/hhai/l20-pretrain
export PATH="$PWD/.venv-continue/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STATE_DIR=runs/cpt10b-curriculum-state
LOG_DIR=logs/cpt10b-curriculum
mkdir -p "$STATE_DIR" "$LOG_DIR"

set_state() {
  printf '{"status":"running","stage":"%s","updated_at":"%s"}\n' \
    "$1" "$(date -Is)" > "$STATE_DIR/status.json"
}

run_stage() {
  local name="$1"
  local config="$2"
  local final_dir="$3"
  if [ -f "$final_dir/config.json" ]; then
    return
  fi
  set_state "$name"
  python -m l20_pretrain.train "$config" \
    2>&1 | tee "$LOG_DIR/$name.log"
}

run_stage phase1_2k configs/l20_edu_135m_cpt10b_2k.yaml runs/l20-edu-135m-cpt10b-2k/final
run_stage phase2_4k configs/l20_edu_135m_cpt10b_4k.yaml runs/l20-edu-135m-cpt10b-4k/final
run_stage phase3_8k configs/l20_edu_135m_cpt10b_8k.yaml runs/l20-edu-135m-cpt10b-8k/final

set_state eval
CANDIDATE=ours-cpt10b OUTPUT_ROOT=eval_results/cpt10b-final \
  bash scripts/eval_smollm_benchmark.sh \
    "ours-cpt10b=runs/l20-edu-135m-cpt10b-8k/final" \
    "smollm-135m=HuggingFaceTB/SmolLM-135M" \
    "smollm2-135m=HuggingFaceTB/SmolLM2-135M" \
    2>&1 | tee "$LOG_DIR/eval.log"

printf '{"status":"complete","stage":"complete","updated_at":"%s"}\n' \
  "$(date -Is)" > "$STATE_DIR/status.json"
