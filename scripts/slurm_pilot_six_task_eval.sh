#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
model_path=${MODEL_PATH:?MODEL_PATH must identify the selected local checkpoint}
model_label=${MODEL_LABEL:-pilot-winner}
output_root=${OUTPUT_ROOT:-$project_root/eval_results/pilot-six-task-${SLURM_JOB_ID:-manual}}

test -s "$model_path/model.safetensors"
test ! -e "$output_root"

eval_site=$(mktemp -d "/tmp/l20-eval-${SLURM_JOB_ID:-manual}.XXXXXX")
cleanup() {
  rm -rf -- "$eval_site"
}
trap cleanup EXIT

"$runtime_python" -m pip install \
  --quiet \
  --no-index \
  --find-links "$project_root/wheels" \
  --target "$eval_site" \
  --no-deps \
  lm-eval==0.4.11 evaluate==0.4.6 jsonlines==4.0.0 \
  pytablewriter==1.2.1 rouge-score==0.1.2 sacrebleu==2.6.0 \
  scikit-learn==1.9.0 sqlitedict==2.1.0 word2number==1.1 \
  more_itertools==11.1.0 absl-py==2.5.0 nltk==3.10.3 six==1.17.0 \
  portalocker==4.3.0 regex==2026.9.3 tabulate==0.10.0 colorama==0.4.6 \
  lxml==6.1.3 joblib==1.6.0 cloudpickle==3.1.2 narwhals==2.25.0 \
  scipy==1.18.1 threadpoolctl==3.6.0 DataProperty==1.1.1 \
  mbstrdecoder==1.1.5 pathvalidate==3.3.1 tabledata==1.3.5 \
  tcolorpy==0.1.7 typepy==1.3.5 chardet==6.0.0.post1 pytz==2026.3.post1

export PYTHONPATH="$eval_site:$overlay:$project_root/src"
export PATH="$eval_site/bin:$PATH"
export PYTHON="$runtime_python"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TASKS=arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,winogrande
export DEVICE=cuda:0
export DTYPE=bfloat16
export BATCH_SIZE=auto
export MAX_BATCH_SIZE=16
export SEED=20260906
export OUTPUT_ROOT="$output_root"
export CANDIDATE="$model_label"

cd "$project_root"
echo "execution_scope=pilot_checkpoint_six_task_not_final_sota"
echo "lm_eval_version=$("$runtime_python" -c 'import importlib.metadata as m; print(m.version("lm-eval"))')"
bash scripts/eval_smollm_benchmark.sh "$model_label=$model_path"
echo "eval_output=$output_root"
echo "six_task_eval_status=pass"
