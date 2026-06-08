# Stage 2 Speed and Quality Plan

This document prepares the next stage after the current 8K continued pretraining run finishes.

## Default Next Run

Use the replay mixture unless an ablation clearly beats it:

```bash
bash scripts/prepare_l20_stage2_math_code_textbook_replay_8k.sh
bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh
```

Default recipe:

- 15% Stage-1 high-quality edu replay from `data/l20_edu_hq_8k`
- 30% FineMath `finemath-4plus`
- 22% Stack-Edu permissive code
- 33% Cosmopedia textbook-style data

Replay is deliberately included to reduce distribution drift during continued pretraining.

## Speed Benchmark

Run this only when the GPU is not occupied by a production training job:

```bash
bash scripts/setup_liger_kernel.sh
bash scripts/benchmark_l20_speed_variants.sh
```

The benchmark compares:

- baseline microbatch sizes
- gradient checkpointing variants
- `torch.compile`
- Liger Kernel
- Liger + compile

Results are written to `docs/l20_speed_benchmark.jsonl`.

Choose the fastest variant that stays comfortably below the L20 memory limit. If Liger only saves memory but does not improve tokens/sec, use the memory headroom to test a larger `micro_batch_size` with lower `gradient_accumulation_steps`.

## Quality Benchmark

Prepare domain validation shards:

```bash
bash scripts/prepare_l20_eval_domains_8k.sh
```

Evaluate a checkpoint:

```bash
bash scripts/eval_l20_domain_ppl_8k.sh runs/l20-edu-135m-hq-longctx-8k/final
```

Track at least these domains:

- general edu
- math
- code
- synthetic textbook

Do not select a Stage-2 data mix by total validation loss alone. A mix that improves math and code while hurting general edu too much should be rejected or given more replay.

## Ablation Ladder

Before scaling to 300M+ Stage-2 tokens, run 50M-token pilots:

- current no-replay recipe
- replay recipe
- math-heavy
- code-heavy
- textbook-heavy
- replay 20%

Select by domain PPL plus downstream evals, not only by training loss.

## References

- Continued pretraining replay and LR schedules: https://arxiv.org/abs/2403.08763
- DCLM data curation and model-based filtering: https://arxiv.org/abs/2406.11794
- FineWeb/FineWeb-Edu filtering: https://arxiv.org/abs/2406.17557
- SmolLM2 data-centric recipe: https://arxiv.org/abs/2502.02737
- Liger Kernel: https://arxiv.org/abs/2410.10989
- PyTorch SDPA: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
