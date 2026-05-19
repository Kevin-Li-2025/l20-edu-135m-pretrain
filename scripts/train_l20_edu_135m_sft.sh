#!/usr/bin/env bash
set -euo pipefail

python -m l20_pretrain.train_sft configs/l20_edu_135m_sft.yaml "$@"
