#!/usr/bin/env bash
set -euo pipefail

cd /home/hhai/l20-pretrain
TRAIN_PATTERN="l20_pretrain.train configs/l20_edu_135m_stage3_current_shard_8k.yaml"

while pgrep -f "$TRAIN_PATTERN" >/dev/null; do
  sleep 300
done

exec bash scripts/prepare_l20_stage4_hq_crossdedup_8k.sh
