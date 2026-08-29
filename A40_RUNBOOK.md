# A40 DDP runbook

This checkout contains the high-throughput path for one node with five or six
48 GB A40 GPUs. It uses pure data parallelism: no tensor, pipeline, or FSDP
sharding is needed for the 135M model.

## Install and data

```bash
python -m pip install -e '.[speed]'
l20-prepare-shards \
  --output-dir data/fineweb_edu_2k \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --dataset HuggingFaceFW/fineweb-edu \
  --config-name sample-100BT \
  --target-tokens 12000000000 \
  --val-tokens 10000000 \
  --block-size 2048
```

The launch configs expect `data/fineweb_edu_2k/train.bin` and optional
`val.bin`. Pretokenization is required for a meaningful throughput run; raw
streaming text makes tokenizer/network speed part of the benchmark.

## Preflight

Inspect topology before training:

```bash
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv
```

Run 30 optimizer updates and ignore the first compiled steps when reading
throughput:

```bash
A40_NPROC_PER_NODE=6 A40_PREFLIGHT_STEPS=30 scripts/train_a40_ddp.sh
```

Acceptance gates after warm-up:

- all ranks remain alive and losses match within the reduced DDP metric;
- no repeated NCCL timeout or P2P error;
- no CUDA OOM at micro-batch 28 (six GPUs) or 32 (five GPUs);
- global `tokens_per_sec_window` is stable over at least three log windows;
- no host-side tokenizer or disk bottleneck in the pretokenized run.

On the measured five-A40 host, the native mixed P2P/SHM ring initialized but
hung on the first collective. The same smoke test passed with
`NCCL_P2P_DISABLE=1`, so `scripts/train_a40_ddp.sh` enables that host-mediated
fallback by default. Override it only after a fresh native-P2P smoke passes.

## Full run

```bash
# Six A40s, 12.0B planned tokens
A40_NPROC_PER_NODE=6 scripts/train_a40_ddp.sh

# Five A40s, 12.0B planned tokens
A40_NPROC_PER_NODE=5 scripts/train_a40_ddp.sh
```

Only rank zero writes logs and checkpoints. Checkpointing and evaluation are
barriered so other ranks cannot race into the next update while rank zero is
writing shared state.
