#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
checkpoint="${CONTINUATION_2B_CHECKPOINT:-$repo_root/runs/a40-5x-smollm2-continuation-2b-1k-repair-lr3e4/step-002035}"
eval_root="${CONTINUATION_2B_EVAL_ROOT:-$repo_root/eval_results/smollm2-continuation-2b-final}"
candidate_eval="$eval_root/candidate"
start_eval="$repo_root/eval_results/smollm2-repro-pretrain-9p8b-broad"
pilot_eval="$repo_root/eval_results/smollm2-continual-pilots/a40-5x-smollm2-continual-repair-500m-1k-lr3e4/eval"
official_root="$repo_root/eval_results/smollm2-repro-pretrain-9p8b-final"
log_file="${CONTINUATION_2B_EVAL_LOG:-$repo_root/a40_smollm2_continuation_2b_eval.log}"
lock_file="${CONTINUATION_2B_EVAL_LOCK:-$repo_root/.smollm2_continuation_2b_eval.lock}"
confidence="${CONTINUATION_2B_CONFIDENCE:-0.9833}"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export TOKENIZERS_PARALLELISM=false

exec 9>"$lock_file"
if ! flock -n 9; then
  printf '%s evaluator already active; exiting\n' "$(date -Is)" >> "$log_file"
  exit 0
fi

mkdir -p "$eval_root"
if [[ -f "$eval_root/.complete" ]]; then
  printf '%s evaluation already complete; exiting\n' "$(date -Is)" >> "$log_file"
  exit 0
fi

for required in config.json model.safetensors tokenizer.json; do
  if [[ ! -s "$checkpoint/$required" ]]; then
    printf '%s missing final checkpoint artifact: %s\n' "$(date -Is)" "$checkpoint/$required" >> "$log_file"
    exit 1
  fi
done
for baseline in "$start_eval" "$pilot_eval" "$official_root/smollm" "$official_root/smollm2"; do
  if [[ ! -f "$baseline/.complete" ]]; then
    printf '%s incomplete baseline evaluation: %s\n' "$(date -Is)" "$baseline" >> "$log_file"
    exit 1
  fi
done

printf '%s generating fixed greedy completions\n' "$(date -Is)" >> "$log_file"
"$python_bin" - "$checkpoint" "$eval_root/generations.json" <<'PY' >> "$log_file" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = sys.argv[1]
output = Path(sys.argv[2])
prompts = [
    "The capital of France is",
    "Water freezes when its temperature reaches",
    "Photosynthesis is the process by which plants",
    "To solve the equation 2x + 3 = 11, first",
    "A healthy way to manage stress is",
    "In Python, a function that returns the square of x can be written as",
    "Once upon a time in a small village,",
    "Artificial intelligence can improve education by",
    "人工智能可以帮助教育，因为",
]
models = {
    "start": "/workspace/a40-pretrain/runs/a40-5x-smollm2-repro-pretrain-9p8b/step-009969",
    "pilot_500m": "/workspace/a40-pretrain/runs/a40-5x-smollm2-continual-repair-500m-1k-lr3e4/step-000510",
    "final_2b": checkpoint,
}
payload = {
    "method": {
        "decoding": "greedy",
        "max_new_tokens": 64,
        "prompts": prompts,
    },
    "models": {},
}
for name, model_path in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()
    rows = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        rows.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ),
            }
        )
    payload["models"][name] = {"checkpoint": model_path, "rows": rows}
    del model
    torch.cuda.empty_cache()

output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"event": "generations_done", "path": str(output)}, ensure_ascii=False))
PY

if [[ ! -f "$candidate_eval/.complete" ]]; then
  staging="$(mktemp -d "$eval_root/.candidate.partial.XXXXXX")"
  printf '%s starting frozen ten-task benchmark\n' "$(date -Is)" >> "$log_file"
  if EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}" \
    EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}" \
    A40_PYTHON="$python_bin" \
    bash scripts/eval_pretrain_broad_parallel.sh "$checkpoint" "$staging" >> "$log_file" 2>&1; then
    mv "$staging" "$candidate_eval"
  else
    failed="${staging}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$staging" "$failed"
    printf '%s benchmark failed: %s\n' "$(date -Is)" "$failed" >> "$log_file"
    exit 1
  fi
fi

paired() {
  local baseline="$1"
  local suffix="$2"
  shift 2
  "$python_bin" -m l20_pretrain.paired_eval \
    --baseline "$baseline" \
    --candidate "$candidate_eval" \
    --confidence "$confidence" \
    --out "$eval_root/paired-$suffix.json" \
    "$@" >> "$log_file" 2>&1
}

paired "$start_eval" "vs-start-primary"
paired "$start_eval" "vs-start-development" \
  --task openbookqa --task sciq --task boolq --task swag
paired "$pilot_eval" "vs-pilot-500m-primary"
paired "$pilot_eval" "vs-pilot-500m-development" \
  --task openbookqa --task sciq --task boolq --task swag
paired "$official_root/smollm" "vs-smollm-primary"
paired "$official_root/smollm2" "vs-smollm2-primary"

"$python_bin" scripts/summarize_smollm_benchmark.py \
  --result "start=$start_eval" \
  --result "pilot_500m=$pilot_eval" \
  --result "smollm=$official_root/smollm" \
  --result "smollm2=$official_root/smollm2" \
  --result "final_2b=$candidate_eval" \
  --candidate final_2b \
  --baseline start \
  --baseline pilot_500m \
  --baseline smollm \
  --baseline smollm2 \
  --out "$eval_root/aggregate-comparison.json" >> "$log_file" 2>&1

"$python_bin" scripts/check_continual_pilot.py \
  --primary-paired "$eval_root/paired-vs-start-primary.json" \
  --development-paired "$eval_root/paired-vs-start-development.json" \
  --min-confidence "$confidence" \
  --out "$eval_root/promotion-vs-start.json" >> "$log_file" 2>&1

touch "$eval_root/.complete"
printf '%s final 2B evaluation complete: %s\n' "$(date -Is)" "$eval_root" >> "$log_file"
