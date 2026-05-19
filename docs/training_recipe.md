# Training Recipe

This document records the exact recipe used for the released
`l20-edu-135m` base checkpoint.

## Model

| Field | Value |
| --- | --- |
| Run name | `l20-edu-135m-deepthin` |
| Parameters | 134,515,008 |
| Architecture | Llama-style decoder-only Transformer |
| Layers | 30 |
| Hidden size | 576 |
| FFN size | 1536 |
| Attention heads | 9 query heads, 3 key/value heads |
| Context length | 2048 |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` |
| Attention implementation | PyTorch SDPA |
| Tied embeddings | yes |

## Data

| Field | Value |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Config | `sample-10BT` |
| Split | `train` |
| Streaming | yes |
| Text filter | `min_chars=300`, `max_chars=50000` |
| Quality filter | `min_score=3.0`, `min_int_score=3` |
| Packing | EOS-joined documents packed into 2048-token blocks |
| Planned token budget | 10,001,252,352 tokens |

## Optimization

| Field | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `4e-4` peak |
| LR schedule | linear warmup + cosine decay to `0.1 * peak_lr` |
| Warmup steps | 1000 |
| Min LR ratio | 0.1 |
| Weight decay | 0.1 |
| Adam beta1 / beta2 | 0.9 / 0.95 |
| Gradient clip | 1.0 |
| Precision | bfloat16 |
| Torch compile | enabled |
| Gradient checkpointing | enabled |

## Batch And Token Accounting

| Field | Value |
| --- | ---: |
| Micro batch size | 6 sequences |
| Gradient accumulation | 43 |
| Global batch size | 258 sequences |
| Sequence length | 2048 tokens |
| Tokens per optimizer step | 528,384 |
| Max steps | 18,928 |
| Planned tokens | 10,001,252,352 |

## Checkpointing And Evaluation

| Field | Value |
| --- | --- |
| Log interval | 10 steps |
| Eval interval | 500 steps |
| Eval batches | 64 |
| Save interval | 1000 steps |
| Checkpoints retained | last 2 |
| Final checkpoint | `runs/l20-edu-135m-deepthin/step-018928` |
| Published checkpoint | `AliceYin/l20-edu-135m` |

Each regular checkpoint represents about 528.4M training tokens. The final
checkpoint was saved at step 18,928 rather than an even 1000-step boundary.

## Runtime And Hardware

| Field | Value |
| --- | --- |
| GPU | NVIDIA L20 |
| Reported GPU memory | 46,068 MiB total |
| Driver | 550.163.01 |
| Mean logged throughput | 38,541 tokens/s |
| Mean logged throughput after step 1000 | 38,587 tokens/s |
| Estimated train time from throughput | about 72.0 hours |
| Final checkpoint mtime | 2026-05-19 05:04:22 +0800 |

The JSON training log did not record an exact wall-clock launch timestamp or
peak VRAM. Peak GPU memory should therefore be treated as **not measured** for
this release. Future runs should log `nvidia-smi --query-gpu=memory.used` during
training.

Cost was not available from billing logs. A reproducible estimate is:

```text
estimated_cost = 72 GPU-hours * L20_hourly_rate
```

Examples:

| L20 Hourly Rate | Estimated GPU Cost |
| ---: | ---: |
| $0.60 / hour | $43 |
| $1.00 / hour | $72 |
| $1.50 / hour | $108 |

This excludes storage, network egress, idle time, and engineering time.

## Known Issues During The Run

- The dataset mirror produced transient `Read timed out` errors near the end of
  training. The run recovered through retry and continued.
- The final perplexity command printed `loss=2.8731 perplexity=17.69`, then hit
  a Python finalization crash. The metric is usable because it was printed before
  process teardown, but the crash is documented.
- After the final checkpoint was written, the training process printed
  `terminate called without an active exception`. The checkpoint was complete
  and load-tested with `AutoModelForCausalLM`.
- Peak VRAM and exact cloud cost were not logged.

## Reproduction Command

```bash
python -m l20_pretrain.train configs/l20_135m_deepthin.yaml
```

The config file is the source of truth for this recipe:
`configs/l20_135m_deepthin.yaml`.
