#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
base_config="configs/a40_5x_smollm2_synthetic_benchmark.yaml"
result_root="${A40_THROUGHPUT_ROOT:-/workspace/a40-pretrain/eval_results/a40-throughput-matrix}"
synthetic_data="${A40_THROUGHPUT_DATA:-/tmp/l20_synthetic_2k}"
mkdir -p "$result_root/configs"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export A40_PYTHON="$python_bin"
export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"

if [[ ! -s "$synthetic_data/train.bin" ]]; then
  "$python_bin" scripts/make_synthetic_tokenized_data.py \
    --output-dir "$synthetic_data" \
    --tokens 64000000 \
    --block-size 2048
fi

write_config() {
  local name="$1" block_size="$2" micro_batch="$3" accumulation="$4"
  local bucket_mb="$5" static_graph="$6" compression="$7"
  local path="$result_root/configs/$name.yaml"
  "$python_bin" - "$base_config" "$path" "$name" "$block_size" "$micro_batch" \
    "$accumulation" "$bucket_mb" "$static_graph" "$compression" "$synthetic_data" <<'PY'
from pathlib import Path
import sys
import yaml

base, output, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
block_size, micro_batch, accumulation = map(int, sys.argv[4:7])
bucket_mb = float(sys.argv[7])
static_graph = sys.argv[8].lower() == "true"
compression, data_path = sys.argv[9], sys.argv[10]
payload = yaml.safe_load(base.read_text(encoding="utf-8"))
payload["run_name"] = f"a40-throughput-{name}"
payload["output_dir"] = f"/tmp/a40-throughput-{name}"
payload["dataset"]["tokenized_path"] = data_path
if name == "wide-2k-5g":
    # Architecture-only ceiling candidate.  Keep parameter count and global
    # tokens/update close to the SmolLM2 baseline, but use fewer, wider layers
    # to expose larger GEMMs on A40.  This is a throughput measurement, not a
    # checkpoint-compatible production recommendation.
    payload["init_model_name_or_path"] = None
    payload["model"].update(
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=15,
        num_attention_heads=12,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
    )
payload["model"]["block_size"] = block_size
trainer = payload["trainer"]
trainer.update(
    micro_batch_size=micro_batch,
    gradient_accumulation_steps=accumulation,
    max_steps=30,
    warmup_steps=1,
    learning_rate=5e-6,
    log_interval=5,
    eval_interval=0,
    eval_batches=0,
    save_interval=0,
    keep_last_checkpoints=1,
    ddp_bucket_cap_mb=bucket_mb,
    ddp_static_graph=static_graph,
    ddp_gradient_compression=compression,
)
if name == "compile-nocg-2k-5g":
    trainer.update(
        compile=True,
        compile_mode="max-autotune-no-cudagraphs",
        compile_fullgraph=False,
        compile_scope="backbone",
    )
output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  printf '%s\n' "$path"
}

run_case() {
  local name="$1" block_size="$2" micro_batch="$3" accumulation="$4"
  local bucket_mb="$5" static_graph="$6" compression="$7"
  local nproc="$8" visible_gpus="$9" nccl_algo="${10}"
  local p2p="${11:-1}" max_connections="${12:-default}"
  local case_dir="$result_root/$name" config
  local -a env_cmd=(env)
  mkdir -p "$case_dir"
  config="$(write_config "$name" "$block_size" "$micro_batch" "$accumulation" \
    "$bucket_mb" "$static_graph" "$compression")"
  if [[ "$nccl_algo" == "default" ]]; then
    env_cmd+=(-u NCCL_ALGO)
  fi
  if [[ "$max_connections" == "default" ]]; then
    env_cmd+=(-u CUDA_DEVICE_MAX_CONNECTIONS)
  fi
  if [[ "$nccl_algo" != "default" ]]; then
    env_cmd+=("NCCL_ALGO=$nccl_algo")
  fi
  if [[ "$max_connections" != "default" ]]; then
    env_cmd+=("CUDA_DEVICE_MAX_CONNECTIONS=$max_connections")
  fi
  env_cmd+=(
    "CUDA_VISIBLE_DEVICES=$visible_gpus"
    "A40_NPROC_PER_NODE=$nproc"
    "NCCL_P2P_DISABLE=$p2p"
  )
  case "$name" in
    affinity1-*) env_cmd+=("NCCL_IGNORE_CPU_AFFINITY=1") ;;
    shmoff-*) env_cmd+=("NCCL_SHM_DISABLE=1") ;;
    nthreads512-*) env_cmd+=("NCCL_NTHREADS=512") ;;
    compile-nocg-*) env_cmd+=("TORCHINDUCTOR_COMPILE_THREADS=4") ;;
  esac

  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
    --format=csv,noheader,nounits \
    -l 1 > "$case_dir/telemetry.csv" 2>&1 &
  local telemetry_pid=$!
  set +e
  timeout "${A40_CASE_TIMEOUT_SEC:-1200}" "${env_cmd[@]}" \
    bash scripts/train_a40_ddp.sh "$config" > "$case_dir/train.log" 2>&1
  local status=$?
  set -e
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  if (( status == 0 )); then
    touch "$case_dir/.complete"
  else
    printf 'exit_status=%s\n' "$status" > "$case_dir/failure.txt"
    touch "$case_dir/.failed"
  fi
}

# All cases process exactly 983,040 tokens/update. This isolates kernel,
# communication, sequence-length, and topology effects without a larger-batch
# shortcut that could reduce optimization efficiency.
run_case baseline-2k-5g       2048 24 4 100 false none 5 0,1,2,3,4 default 1 default
run_case static100-2k-5g      2048 24 4 100 true  none 5 0,1,2,3,4 default 1 default
run_case bucket25-2k-5g       2048 24 4  25 false none 5 0,1,2,3,4 default 1 default
run_case static25-2k-5g       2048 24 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case bf16comm-2k-5g       2048 24 4  25 true  bf16 5 0,1,2,3,4 default 1 default
run_case short-1k-5g          1024 48 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case short-512-5g          512 96 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case ring-2k-5g           2048 24 4  25 true  none 5 0,1,2,3,4 Ring    1 default
run_case tree-2k-5g           2048 24 4  25 true  none 5 0,1,2,3,4 Tree    1 default
run_case connections1-2k-5g  2048 24 4  25 true  none 5 0,1,2,3,4 default 1 1
run_case affinity1-2k-5g     2048 24 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case shmoff-2k-5g        2048 24 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case nthreads512-2k-5g   2048 24 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case numa1-2k-4g          2048 24 5  25 true  none 4 1,2,3,4   default 1 default
run_case numa1-p2p-2k-4g      2048 24 5  25 true  none 4 1,2,3,4   default 0 default
run_case compile-nocg-2k-5g  2048 24 4  25 true  none 5 0,1,2,3,4 default 1 default
run_case wide-2k-5g           2048 32 3  25 true  none 5 0,1,2,3,4 default 1 default

"$python_bin" scripts/summarize_throughput_matrix.py "$result_root" \
  --min-step 10 \
  --out "$result_root/summary.json"
