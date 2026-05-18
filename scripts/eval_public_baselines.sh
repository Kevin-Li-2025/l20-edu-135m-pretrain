#!/usr/bin/env bash
set -euo pipefail

scripts/eval_lm_harness.sh openai-community/gpt2 eval_results/gpt2-small
scripts/eval_lm_harness.sh facebook/opt-125m eval_results/opt-125m
scripts/eval_lm_harness.sh EleutherAI/gpt-neo-125m eval_results/gpt-neo-125m
scripts/eval_lm_harness.sh cerebras/Cerebras-GPT-111M eval_results/cerebras-gpt-111m
scripts/eval_lm_harness.sh EleutherAI/pythia-160m eval_results/pythia-160m
scripts/eval_lm_harness.sh HuggingFaceTB/SmolLM-135M eval_results/smollm-135m
scripts/eval_lm_harness.sh HuggingFaceTB/SmolLM2-135M eval_results/smollm2-135m
