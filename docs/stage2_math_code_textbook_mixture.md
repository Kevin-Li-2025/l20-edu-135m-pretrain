# Stage 2 Math/Code/Synthetic Textbook Mixture

This stage is intended to run after the 8K FineWeb-Edu continued pretraining checkpoint is complete.

## Goal

Improve reasoning-style educational behavior without turning the 135M model into a chat model yet. This is still causal language-model continued pretraining, not SFT.

## Default Size

- Train tokens: 300,000,000
- Validation tokens: 2,097,152
- Sequence length: 8192
- Steps: 555 at 540,672 tokens/step

## Recommended Variant

Prefer the replay recipe for the next full Stage-2 run unless a pilot ablation beats it:

```bash
bash scripts/prepare_l20_stage2_math_code_textbook_replay_8k.sh
bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh
```

That variant keeps 15% Stage-1 high-quality edu replay and rebalances the new-domain mix to reduce drift during continued pretraining.

## Data Mix

- 35% `HuggingFaceTB/finemath`, config `finemath-4plus`
- 25% `HuggingFaceTB/stack-edu`
  - Python 8%
  - JavaScript 4%
  - TypeScript 3%
  - C++ 3%
  - Rust 2.5%
  - SQL 2.5%
  - Shell 2%
  - Only `license_type=permissive`
  - Downloads code contents from Software Heritage S3 by `blob_id`
- 40% `HuggingFaceTB/cosmopedia`
  - OpenStax 16%
  - Khan Academy 12%
  - AutoMathText (`auto_math_text`) 12%

## Why These Sources

- FineMath is the math pretraining source used by the SmolLM2 data-centric recipe.
- Stack-Edu is educational code filtered from StarCoder2Data, also used by SmolLM2.
- Cosmopedia provides synthetic textbook-style material inspired by the Phi/Textbooks Are All You Need direction.

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
- Cosmopedia: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- SmolLM2: https://arxiv.org/abs/2502.02737
- FineWeb/FineWeb-Edu: https://arxiv.org/abs/2406.17557
