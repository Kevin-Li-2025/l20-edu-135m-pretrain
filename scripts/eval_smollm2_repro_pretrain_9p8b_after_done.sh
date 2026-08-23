#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
main_log="${SMOLLM2_LONGRUN_LOG:-$repo_root/a40_smollm2_repro_pretrain_9p8b_run.log}"
run_dir="${SMOLLM2_LONGRUN_DIR:-$repo_root/runs/a40-5x-smollm2-repro-pretrain-9p8b}"
eval_root="${SMOLLM2_LONGRUN_EVAL_DIR:-$repo_root/eval_results/smollm2-repro-pretrain-9p8b-final}"
posteval_log="${SMOLLM2_POSTEVAL_LOG:-$repo_root/a40_smollm2_repro_pretrain_9p8b_posteval.log}"
lock_file="${SMOLLM2_POSTEVAL_LOCK:-$repo_root/.smollm2_repro_pretrain_9p8b_posteval.lock}"
poll_seconds="${SMOLLM2_POSTEVAL_POLL_SECONDS:-60}"
training_pattern="/opt/a40-pretrain-venv/bin/python -u -m l20_pretrain.train configs/a40_5x_smollm2_repro_pretrain_9p8b.yaml$"

exec 9>"$lock_file"
if ! flock -n 9; then
  printf '%s post-training evaluator already active; exiting\n' "$(date -Is)" >> "$posteval_log"
  exit 0
fi

mkdir -p "$eval_root"
if [[ -f "$eval_root/.complete" ]]; then
  printf '%s post-training evaluation already complete; exiting\n' "$(date -Is)" >> "$posteval_log"
  exit 0
fi

printf '%s waiting for completed long-run checkpoint\n' "$(date -Is)" >> "$posteval_log"
until [[ -s "$main_log" ]] && grep -q '"event": "done"' "$main_log"; do
  sleep "$poll_seconds"
done
while pgrep -f -- "$training_pattern" >/dev/null; do
  sleep 10
done

checkpoint="$(readlink -f "$run_dir/final")"
for required in config.json model.safetensors tokenizer.json trainer_state.pt; do
  if [[ ! -s "$checkpoint/$required" ]]; then
    printf '%s missing checkpoint artifact: %s\n' "$(date -Is)" "$checkpoint/$required" >> "$posteval_log"
    exit 1
  fi
done

run_eval() {
  local name="$1"
  local model="$2"
  local final_dir="$eval_root/$name"
  local attempt staging timestamp
  if [[ -f "$final_dir/.complete" ]]; then
    return 0
  fi
  for attempt in 1 2 3; do
    staging="$(mktemp -d "$eval_root/.${name}.partial.XXXXXX")"
    printf '%s evaluating %s (attempt %s/3)\n' "$(date -Is)" "$name" "$attempt" >> "$posteval_log"
    if EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}" \
      EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}" \
      A40_PYTHON="$python_bin" \
      bash scripts/eval_smollm_parallel.sh "$model" "$staging" >> "$posteval_log" 2>&1; then
      touch "$staging/.complete"
      if [[ -e "$final_dir" ]]; then
        timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
        mv "$final_dir" "${final_dir}.incomplete.${timestamp}"
      fi
      mv "$staging" "$final_dir"
      return 0
    fi
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$staging" "${final_dir}.failed-${attempt}.${timestamp}"
  done
  printf '%s evaluation failed after three attempts: %s\n' "$(date -Is)" "$name" >> "$posteval_log"
  return 1
}

# Re-run both public baselines with identical harness version, seeds, task
# definitions, dtype, and sample logging. This makes paired significance tests
# possible and avoids comparing against stale aggregate-only result files.
run_eval "candidate" "$checkpoint"
run_eval "smollm2" "HuggingFaceTB/SmolLM2-135M"
run_eval "smollm" "HuggingFaceTB/SmolLM-135M"

"$python_bin" scripts/summarize_smollm_benchmark.py \
  --result "smollm=$eval_root/smollm" \
  --result "smollm2=$eval_root/smollm2" \
  --result "candidate=$eval_root/candidate" \
  --candidate candidate \
  --baseline smollm \
  --baseline smollm2 \
  --out "$eval_root/aggregate-comparison.json" >> "$posteval_log" 2>&1

"$python_bin" -m l20_pretrain.paired_eval \
  --baseline "$eval_root/smollm" \
  --candidate "$eval_root/candidate" \
  --confidence 0.975 \
  --out "$eval_root/paired-vs-smollm.json" >> "$posteval_log" 2>&1
"$python_bin" -m l20_pretrain.paired_eval \
  --baseline "$eval_root/smollm2" \
  --candidate "$eval_root/candidate" \
  --confidence 0.975 \
  --out "$eval_root/paired-vs-smollm2.json" >> "$posteval_log" 2>&1

"$python_bin" scripts/check_smollm_promotion.py \
  --aggregate "$eval_root/aggregate-comparison.json" \
  --paired "smollm=$eval_root/paired-vs-smollm.json" \
  --paired "smollm2=$eval_root/paired-vs-smollm2.json" \
  --min-confidence 0.975 \
  --out "$eval_root/promotion-gate.json" >> "$posteval_log" 2>&1

touch "$eval_root/.complete"
printf '%s post-training six-task evaluation complete: %s\n' \
  "$(date -Is)" "$eval_root" >> "$posteval_log"
