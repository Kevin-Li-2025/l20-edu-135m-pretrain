# A40 maximum-efficiency research plan

Date: 2026-08-22

## Objective and current evidence

The optimization target is lexicographic, not a single raw throughput number:

1. pass the frozen quality gate at equal training tokens;
2. among passing runs, maximize end-to-end tokens/second;
3. preserve exact resume, data provenance, and checkpoint reproducibility.

The completed 5xA40 baseline is 134.515M parameters, 2,048-token sequences,
BF16 autocast with FP32 parameters, fused AdamW, full Liger, PyTorch SDPA, and
DDP with `no_sync()` accumulation. It sustained about 131.7k token/s at 21.67%
estimated MFU with 983,040 global tokens/update. The observed host has no NVLink
and requires `NCCL_P2P_DISABLE=1`; these settings are not portable to a new host.

The completed 9.8B-token checkpoint has a frozen six-task mean near 0.423. It is
above the earlier `AliceYin/l20-edu-135m` card result, but remains materially
below the released SmolLM and SmolLM2 baselines. A configuration change is not
treated as a capability improvement until the paired evaluation gate passes.

## Active controlled experiments

All continual-pretraining arms start from the same step-009969 checkpoint and
consume 501,350,400 tokens with the same 983,040-token global batch:

- original-mixture control, LR 3e-4, 2K sequences;
- repair mixture, LR 3e-4, 2K sequences;
- repair mixture, LR 6e-4, 2K sequences;
- repair mixture, LR 3e-4, 1K sequences with twice as many sequences per batch.

The repair mixture is approximately 60% FineWeb-Edu, 26% DCLM-Edu, 5%
Stack-Edu, 4% InfiMM-WebMath, 3% FineMath, and 2% Cosmopedia. The 1K arm is an
equal-token sequence-length test, not a claim that shorter context is always
better.

Each arm is evaluated on the frozen ARC-Challenge, ARC-Easy, HellaSwag,
LAMBADA, PIQA, and WinoGrande promotion set plus OpenBookQA, SciQ, BoolQ, and
SWAG development tasks. Promotion requires a positive paired-bootstrap lower
bound, positive point deltas on ARC-Challenge/ARC-Easy/HellaSwag, a nonnegative
development mean, and no task regression larger than 0.5 percentage points.

## Host throughput matrix

`scripts/run_a40_throughput_matrix.sh` benchmarks seventeen equal-token cases after
the quality pilots finish:

- current 2K/5-GPU baseline;
- 25/100 MiB DDP buckets with and without static graph;
- BF16 gradient communication;
- 1K and 512 sequence lengths at constant tokens/update;
- NCCL default, Ring, and Tree selection;
- the official SmolLM-style `CUDA_DEVICE_MAX_CONNECTIONS=1` setting;
- NCCL GPU-aware CPU affinity, shared-memory bypass, and communication threads;
- four same-NUMA GPUs with host transport and native P2P versus all five GPUs.
- TorchInductor `max-autotune-no-cudagraphs` on the backbone, which directly
  tests whether compilation can help after removing the previously failing
  CUDA Graph path.
- a 132.144M-parameter 15-layer/768-hidden architecture ceiling at the same
  983,040 tokens/update. An earlier eight-step run measured about 182.2k
  token/s and 26.19% MFU versus 131.7k token/s and 21.67% MFU for the exact
  SmolLM2 architecture. It is not checkpoint-compatible and is not eligible
  for production without a separate equal-token capability study.

Earlier compiled-backbone tests on this exact model produced non-finite
gradients or CUDA Graph overwritten-tensor failures. The matrix therefore tests
only `max-autotune-no-cudagraphs`; a failed or numerically divergent result is
rejected rather than retried in production.
It also excludes FP8 because A40 provides BF16/FP16 Tensor Cores but no native
FP8 Tensor Core path. Micro-batch 32 at 2K is excluded because it crossed the
observed memory limit.

BF16 gradient communication can only be promoted after a quality pilot because
it changes reduction numerics. DDP bucket size, static graph, and NCCL algorithm
are selected by measured steady-state throughput on this host. Each case runs
30 updates and is ranked on five post-warmup windows; the summary records both
median token/s and achieved estimated TFLOP/s so that a lower-FLOP architecture
cannot masquerade as a pure systems speedup.

## Production continuation candidate

The long run is deliberately not auto-launched. If a pilot passes, the initial
5B continuation candidate is:

- exact winning checkpoint/data/LR arm;
- 5 GPUs, 983,040 global tokens/update;
- 5,086 updates = 4,999,741,440 tokens;
- 5% warmup, 75% stable phase, 20% linear decay to zero;
- AdamW beta1=0.9, beta2=0.95, weight decay 0.01, clip norm 1.0;
- BF16 autocast, FP32 parameters, fused AdamW, full Liger, SDPA;
- DDP/NCCL settings chosen by the host matrix;
- checkpoints about every 1B tokens and PPL checks about every 250M tokens.

If no arm passes, more tokens are not launched from a failed recipe. The next
step is a targeted mixture or optimizer ablation, not an unsupported long run.

## Research decisions

- Variable sequence-length curricula have published evidence for reducing
  attention cost and reaching target quality faster, but the reported gains use
  length-bucketed data and cannot be transferred blindly to fixed packed data.
- DoReMi and RegMix show that data mixture can dominate token efficiency and
  that intuitive mixtures can be wrong. The current repair mixture is therefore
  a pilot, not a final optimum.
- Muon has strong from-scratch pretraining evidence, including large-batch data
  efficiency, but changing optimizer on an AdamW-pretrained continuation is a
  different regime. It is reserved for a separate from-scratch ablation.
- MobileLLM supports deep/thin, tied-embedding, GQA designs at this scale; the
  current SmolLM2 architecture already incorporates those choices. No compatible
  structural edit can be promised to improve an existing checkpoint.
- SmolLM3's NoPE, intra-document masking, and embedding-weight-decay changes
  were ablated at 3B/100B-token scale. They are candidates for a new architecture,
  not untested changes to this 135M continuation.

## Primary references

- SmolLM2: https://arxiv.org/abs/2502.02737
- SmolLM/135M recipe: https://huggingface.co/blog/smollm
- SmolLM3 recipe: https://huggingface.co/blog/smollm3
- MobileLLM: https://arxiv.org/abs/2402.14905
- Dataset Decomposition / variable sequence length: https://arxiv.org/abs/2405.13226
- Critical Batch Size Revisited: https://arxiv.org/abs/2505.23971
- DoReMi: https://arxiv.org/abs/2305.10429
- RegMix: https://arxiv.org/abs/2407.01492
- Muon practical efficiency: https://arxiv.org/abs/2505.02222
- Liger Kernel: https://arxiv.org/abs/2410.10989
- PyTorch DDP communication hooks: https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html
- NVIDIA A40 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf
