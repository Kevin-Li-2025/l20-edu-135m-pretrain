# FineWeb-Edu 1B-token factorial

This bundle records the September 6, 2026 scale-up of the matched 50M-token
architecture-by-schedule pilots to a one-billion-token budget. The launcher
requested one RTX 4090 for each independent array cell.

## Outcome

| Architecture | Parameters | Schedule | State | Final validation loss | Perplexity | Median tok/s | Median MFU |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Deep-thin | 134,515,008 | Cosine | OOM at step 1 | - | - | - | - |
| Deep-thin | 134,515,008 | WSD | OOM at step 1 | - | - | - | - |
| Wide | 141,576,960 | Cosine | Completed | 3.2223 | 25.0864 | 69,239 | 90.21% |
| Wide | 141,576,960 | WSD | Completed | **3.1545** | **23.4402** | 69,263 | 90.24% |

For the completed wide pair, WSD reduced endpoint validation loss by 0.06787
and perplexity by 6.56% relative to cosine. Throughput was effectively
unchanged. This repeats the direction seen in the preceding 50M-token pilot,
where the wide WSD cell also had the lowest validation loss.

## What this establishes

The useful result is a matched, single-seed schedule ablation for the 141.6M
wide architecture. Both cells consumed 999,997,440 training tokens from the
same hash-verified FineWeb-Edu shard set and evaluated against the same distinct
validation shard. Both were single-GPU BF16 runs with compilation and gradient
checkpointing disabled.

The result does **not** complete the planned four-cell factorial. The 134.5M
deep-thin architecture exceeded the roughly 22 GiB available device memory in
both schedule cells during the first backward pass. Those failures remain in
the machine-readable receipt rather than being dropped from the comparison.

One seed, non-deterministic CUDA kernels, and separate GPU nodes are not enough
for a statistical schedule-superiority claim. No downstream benchmark was run
from these checkpoints. The result therefore informs architecture and schedule
selection; it does not replace the repository's final-model evaluation.

## Evidence

[`factorial_20260906.json`](factorial_20260906.json) records:

- the four resolved Slurm jobs and terminal states;
- exact dataset, tokenizer, config, source, log, telemetry, and checkpoint
  SHA256 values;
- checkpoint byte sizes without committing the weights;
- steady-state throughput and MFU summaries; and
- explicit supported and unsupported claims.

Validate the committed receipt and its repository-local source/config digests:

```bash
python scripts/verify_fineweb_1b_result.py \
  results/fineweb_1b/factorial_20260906.json
```

Raw logs, telemetry streams, packed token shards, and checkpoints remain on the
training host. They are deliberately excluded from normal Git history.
