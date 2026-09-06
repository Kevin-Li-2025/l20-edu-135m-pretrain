#!/usr/bin/env bash
set -euo pipefail
project_root=${PROJECT_ROOT:?frozen recovery project required}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
shared_root=/ssd/scxi253/l20-edu-135m-v2-dev
index=${SLURM_ARRAY_TASK_ID:?array index required}
roles=(deep_cosine wide_cosine)
if [[ "$index" != 0 && "$index" != 1 ]]; then exit 2; fi
export PYTHONPATH="$shared_root/runtime-overlay:$project_root/src"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$shared_root/hf-cache" TMPDIR="$project_root/tmp"
cd "$project_root"
sha256sum -c EXECUTION_SHA256SUMS
"$runtime_python" scripts/verify_shard_manifest.py data/v2_fineweb_1b
"$runtime_python" scripts/probe_fineweb_memory.py \
  "configs/fineweb_recovery/${roles[$index]}_s20260906.yaml" \
  --output "probe/${roles[$index]}-${SLURM_JOB_ID}.json"
