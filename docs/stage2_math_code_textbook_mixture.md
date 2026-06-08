# Stage 2 Math/Code/Synthetic Textbook Mixture

This stage is intended to run after the 8K FineWeb-Edu continued pretraining checkpoint is complete.

## Goal

Improve reasoning-style educational behavior without turning the 135M model into a chat model yet. This is still causal language-model continued pretraining, not SFT.

## Default Size

- Train tokens: 1,000,000,000
- Validation tokens: 4,194,304
- Sequence length: 8192
- Steps: 1,850 at 540,672 tokens/step

## Recommended Variant

Prefer the replay recipe for the next full Stage-2 run unless a pilot ablation beats it:

```bash
bash scripts/prepare_l20_stage2_math_code_textbook_replay_8k.sh
bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh
```

That variant keeps 20% Stage-1 high-quality edu replay and rebalances the new-domain mix to reduce drift during continued pretraining.

## Data Mix

- 20% tokenized replay from `data/l20_edu_hq_8k`
- 35% `HuggingFaceTB/smollm-corpus`, config `fineweb-edu-dedup`, score >= 4
- 20% `HuggingFaceTB/finemath`, config `finemath-4plus`
- 20% `HuggingFaceTB/stack-edu`
  - Python 8%
  - JavaScript 3.5%
  - TypeScript 2.5%
  - C++ 2.5%
  - Rust 1.5%
  - SQL 1.25%
  - Shell 0.75%
  - Only `license_type=permissive`
  - Downloads code contents from Software Heritage S3 by `blob_id`
- 5% `HuggingFaceTB/smollm-corpus`, config `cosmopedia-v2`

## Why These Sources

- SmolLM-Corpus FineWeb-Edu-Dedup is a 220B-token deduplicated educational web subset designed for small language models.
- FineMath is the math pretraining source used by the SmolLM2 data-centric recipe.
- Stack-Edu is educational code filtered from StarCoder2Data, also used by SmolLM2.
- Cosmopedia v2 provides synthetic textbook-style material, but is intentionally capped at 5% to limit repetitive generated style.

## Commands

Prepare shards:

```bash
bash scripts/prepare_l20_stage2_math_code_textbook_8k.sh
```

Train after stage 1 has finished:

```bash
bash scripts/train_l20_stage2_math_code_textbook_8k.sh
```

Override size for a smaller pilot:

```bash
TARGET_TOKENS=50000000 VAL_TOKENS=1048576 bash scripts/prepare_l20_stage2_math_code_textbook_8k.sh
```

## References

- FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath
- Stack-Edu: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- SmolLM2: https://arxiv.org/abs/2502.02737
- FineWeb/FineWeb-Edu: https://arxiv.org/abs/2406.17557
