# Stage 2 Speed and Quality Plan

This document prepares the next stage after the current 8K continued pretraining run finishes.

## Default Next Run

Use the replay mixture unless an ablation clearly beats it:

```bash
bash scripts/prepare_l20_stage2_math_code_textbook_replay_8k.sh
bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh
```

Default recipe:

- 20% Stage-1 high-quality edu replay from `data/l20_edu_hq_8k`
- 35% SmolLM-Corpus FineWeb-Edu-Dedup, score >= 4
- 20% FineMath `finemath-4plus`
- 20% Stack-Edu permissive code
- 5% SmolLM-Corpus Cosmopedia v2 textbook-style data

This run is now sized at 1,000,000,000 train tokens plus 4,194,304 validation tokens. Replay and deduplicated educational web are deliberately kept as more than half the mixture to reduce distribution drift during continued pretraining. Synthetic textbook data is kept small because the current step-400 probe already showed repetition; the goal is to use it as high-density style/curriculum data, not as the dominant distribution.

## Why This Mix

This follows the common pattern in strong recent recipes:

- SmolLM2: multi-stage small-model training with web text plus specialized math/code data, selected by ablations.
- DCLM/DataComp-LM: data curation, filtering, and mixing matter as much as raw scale; model-based filtering is especially important.
- FineWeb/FineWeb-Edu: strong web data comes from aggressive filtering and educational-quality scoring.
- Qwen2.5-Coder: code-focused training preserves text and math in the mixture instead of training on code alone.

For this 135M model, the practical version is high-quality edu web/replay as the base, math/code as capability injectors, and small synthetic textbook exposure to avoid repetitive generated style. A strict Chinchilla-style from-scratch target would be roughly 2.7B tokens for 135M parameters, while modern small models such as SmolLM2 deliberately overtrain far beyond that; the 1B Stage-2 target is the largest practical single-L20 step here without turning preparation and training into a multi-day job.

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

Choose the fastest variant that stays comfortably below the L20 memory limit. If Liger only saves memory but does not improve tokens/sec, use the memory headroom to test a larger `micro_batch_size` with lower `gradient_accumulation_steps`. Do not enable Liger or compile for a production run until the benchmark beats the baseline on this exact L20 environment.

Latest L20 short benchmark for the Stage-2 config:

- baseline microbatch 2: 17.8k tok/s, 29.3GB allocated
- baseline microbatch 3: 16.3k tok/s, 43.6GB allocated
- baseline microbatch 4: OOM
- Liger microbatch 2: 19.4k tok/s, 17.0GB allocated
- Liger microbatch 3: 20.2k tok/s, 25.0GB allocated
- Liger microbatch 4: 20.1k tok/s, 33.0GB allocated

Production Stage-2 should use Liger microbatch 3 with gradient accumulation 22. This preserves the same 540,672 tokens/step as the previous microbatch-2 setup while improving measured throughput and leaving large memory headroom.

At 540,672 tokens/step, the 1B-token Stage-2 run uses 1,850 optimizer steps. The latest Liger microbatch-3 benchmark measured about 20.2k train tokens/sec on this L20, so pure training time is roughly 14 hours before eval/checkpoint overhead.

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

Before scaling future 2B+ Stage-2 tokens, run 50M-token pilots:

- current no-replay recipe
- elite replay recipe
- math-heavy
- code-heavy
- textbook-heavy
- replay 20%

Select by domain PPL plus downstream evals, not only by training loss.

## References

- Continued pretraining replay and LR schedules: https://arxiv.org/abs/2403.08763
- DCLM data curation and model-based filtering: https://arxiv.org/abs/2406.11794
- FineWeb/FineWeb-Edu filtering: https://arxiv.org/abs/2406.17557
- SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- DeepSeek-Coder technical report: https://arxiv.org/abs/2401.14196
- DeepSeekMath technical report: https://arxiv.org/abs/2402.03300
- SmolLM2 data-centric recipe: https://arxiv.org/abs/2502.02737
- Qwen2.5 technical report: https://arxiv.org/abs/2412.15115
- Qwen2.5-Coder technical report: https://arxiv.org/abs/2409.12186
- Liger Kernel: https://arxiv.org/abs/2410.10989
- PyTorch SDPA: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
