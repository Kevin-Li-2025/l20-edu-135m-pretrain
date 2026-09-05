#!/usr/bin/env bash
set -euo pipefail

runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
if [[ ! -x "$runtime_python" ]]; then
  echo "runtime Python is missing or not executable: $runtime_python" >&2
  exit 2
fi

"$runtime_python" - <<'PY'
import importlib
import json
import sys

modules = {}
for name in ("torch", "transformers", "datasets", "numpy", "yaml", "pytest"):
    try:
        module = importlib.import_module(name)
        modules[name] = getattr(module, "__version__", "present")
    except Exception as exc:
        modules[name] = f"missing:{type(exc).__name__}:{exc}"

import torch

payload = {
    "python": sys.version,
    "modules": modules,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["cuda_available"] or not payload["bf16_supported"]:
    raise SystemExit(3)
for required in ("transformers", "datasets", "numpy", "yaml", "pytest"):
    if str(modules[required]).startswith("missing:"):
        raise SystemExit(4)
PY
