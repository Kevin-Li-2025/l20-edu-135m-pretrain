#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
BASELINE_JSONL="${BASELINE_JSONL:-eval_results/rlvr/gsm8k_before.jsonl}"
C320_JSONL="${C320_JSONL:-eval_results/rlvr/gsm8k_c320_after.jsonl}"
C320_SUMMARY="${C320_SUMMARY:-eval_results/rlvr/gsm8k_c320_after_summary.json}"
C320_MODEL="${C320_MODEL:-runs/l20-edu-135m-rlvr-gsm8k-grpo-c320/final}"
WARMUP_MODEL="${WARMUP_MODEL:-runs/l20-edu-135m-gsm8k-cot-warmup-rlvr-c320/final}"
RUN_WARMUP_ON_NO_IMPROVE="${RUN_WARMUP_ON_NO_IMPROVE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/chat}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" eval_results/rlvr "$OUTPUT_ROOT"

while [ ! -s "$C320_SUMMARY" ]; do
  sleep 60
done

while pgrep -af "scripts/eval_gsm8k_exact.py|scripts/train_rlvr_gsm8k_grpo.py|scripts/run_rlvr_gsm8k_135m.sh" | grep -v grep >/dev/null; do
  sleep 60
done

results=(--result "baseline=$BASELINE_JSONL")
if [ -s eval_results/rlvr/gsm8k_after.jsonl ]; then
  results+=(--result "rlvr192=eval_results/rlvr/gsm8k_after.jsonl")
fi
results+=(--result "c320=$C320_JSONL")

"$PYTHON" scripts/summarize_rlvr_gsm8k_results.py \
  "${results[@]}" \
  --out-json eval_results/rlvr/gsm8k_rlvr_diagnostics.json \
  --out-md eval_results/rlvr/gsm8k_rlvr_diagnostics.md

decision="$("$PYTHON" - <<'PY'
import json
from pathlib import Path

baseline = json.loads(Path("eval_results/rlvr/gsm8k_before_summary.json").read_text())["accuracy"]
c320 = json.loads(Path("eval_results/rlvr/gsm8k_c320_after_summary.json").read_text())["accuracy"]
payload = {
    "baseline_accuracy": baseline,
    "c320_accuracy": c320,
    "improved": c320 > baseline,
    "delta": c320 - baseline,
}
Path("eval_results/rlvr/gsm8k_c320_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("improved" if payload["improved"] else "no_improve")
PY
)"

if [ "$decision" = "improved" ]; then
  bash scripts/run_chat_model_eval.sh "$C320_MODEL" "$OUTPUT_ROOT/rlvr-c320" \
    2>&1 | tee "$LOG_DIR/chat_eval_rlvr_c320.log"
elif [ "$RUN_WARMUP_ON_NO_IMPROVE" = "1" ]; then
  bash scripts/run_gsm8k_cot_warmup_then_rlvr.sh \
    2>&1 | tee "$LOG_DIR/gsm8k_cot_warmup_then_rlvr.auto.log"
  bash scripts/run_chat_model_eval.sh "$WARMUP_MODEL" "$OUTPUT_ROOT/gsm8k-cot-warmup-rlvr-c320" \
    2>&1 | tee "$LOG_DIR/chat_eval_gsm8k_cot_warmup_rlvr_c320.log"
else
  bash scripts/run_chat_model_eval.sh "$C320_MODEL" "$OUTPUT_ROOT/rlvr-c320-no-improve" \
    2>&1 | tee "$LOG_DIR/chat_eval_rlvr_c320_no_improve.log"
fi
