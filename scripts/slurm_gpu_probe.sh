#!/usr/bin/env bash
set -euo pipefail

echo "hostname=$(hostname)"
nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader

echo "python_candidates"
command -v python3 || true
command -v conda || true
find /opt -maxdepth 3 -type f -name python3 2>/dev/null | head -20 || true

echo "runtime_environment"
env | grep -E '^(CUDA|PATH|LD_LIBRARY_PATH)=' | sort || true
