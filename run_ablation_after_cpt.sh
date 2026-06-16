#!/usr/bin/env bash
set -euo pipefail

cd /home/hhai/l20-pretrain
export PATH="$PWD/.venv-continue/bin:$PWD/.venv-eval/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

LOG_DIR=logs/project_ablation
OUT_DIR=eval_results/project_ablation
mkdir -p "$LOG_DIR" "$OUT_DIR"

active_train() {
  pgrep -af "python -m l20_pretrain.train( |$)|python -m l20_pretrain.train_sft( |$)" || true
}

while active_train | grep -q .; do
  echo "[$(date -Is)] waiting for active training before ablations"
  active_train
  sleep 600
done

if command -v lm_eval >/dev/null 2>&1; then
  CANDIDATE=ours-final OUTPUT_ROOT="$OUT_DIR/final_vs_checkpoints" \
    bash scripts/eval_smollm_benchmark.sh \
      "stage2=runs/l20-edu-135m-stage2-math-code-textbook-replay-8k/step-001850" \
      "stage4-base=runs/l20-edu-135m-stage4-hq-crossdedup-8k/step-002500" \
      "stage4-sft-a0875=runs/l20-edu-135m-stage4-sft-anti-forgetting/interpolated/a0875" \
      "cpt10b-final=runs/l20-edu-135m-cpt10b-8k/final" \
      2>&1 | tee "$LOG_DIR/checkpoint_ablation_eval.log"
else
  echo "lm_eval not available; skipping GPU benchmark ablations" | tee "$LOG_DIR/checkpoint_ablation_eval.log"
fi

python scripts/build_project_report.py 2>&1 | tee "$LOG_DIR/rebuild_report.log"
