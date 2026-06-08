# 8K Continued Pretraining Notes

## Current recipe

- Base checkpoint: `AliceYin/l20-edu-135m`
- Context extension: 2048 -> 8192 with YaRN RoPE scaling
- Data: FineWeb-Edu `sample-10BT`, score >= 3, offline cleaned/deduplicated/tokenized shards
- Train tokens available: 700,000,920
- Validation tokens: 4,195,484
- Training shape: 8192 sequence length, micro batch 2, gradient accumulation 33
- Planned tokens: 699,629,568

## Why MFU is not close to large-model numbers

The L20 is saturated at the NVML level, but MFU is limited by end-to-end model FLOPs efficiency:

- The model is small: 134.5M parameters with hidden size 576. Megatron-LM notes that larger GEMMs raise arithmetic intensity and improve MFU.
- 8K attention increases memory traffic and non-GEMM work.
- Full activation checkpointing reduced memory but added forward recomputation during backward. PyTorch documents this as a speed/memory tradeoff.
- Eager-mode framework overhead and unfused elementwise/norm/MLP pieces matter more for a small model than for multi-billion-parameter models.

## Applied optimizations

- Replaced streaming training input with local mmap token shards to remove Python/HF streaming from the hot path.
- Enabled BF16, TF32, PyTorch SDPA flash/memory-efficient kernels, fused AdamW, and pinned host-to-device transfer.
- Benchmarked checkpointing vs no-checkpointing.
- Switched from `micro_batch_size=6, grad_accum=11, gradient_checkpointing=true` to
  `micro_batch_size=2, grad_accum=33, gradient_checkpointing=false`.
- Kept the same `540,672 tokens/step`, while reducing recompute overhead.

## References

- PyTorch SDPA and `torch.compile` transformer building blocks:
  https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
- PyTorch training throughput and MFU with `torch.compile`:
  https://pytorch.org/blog/maximizing-training-throughput/
- PyTorch activation checkpointing speed/memory tradeoff:
  https://pytorch.org/blog/activation-checkpointing-techniques/
- Megatron-LM performance note on larger GEMMs improving MFU:
  https://github.com/NVIDIA/Megatron-LM
- DataComp-LM on model-based filtering and data curation:
  https://arxiv.org/abs/2406.11794
- Chinchilla compute/data scaling:
  https://arxiv.org/abs/2203.15556
