# FineWeb memory-matched recovery: frozen before new training

This is a new experiment, not a rewrite or continuation of the failed four-cell
record. Old completed checkpoints and pinned source/config receipts stay intact.

- Four cells per seed: 134.5M deep-thin and 141.6M wide, each cosine and WSD.
- Seeds: 20260906, 20260907, 20260908; all outcomes including OOM are retained.
- All 12 cells use microbatch 2, accumulation 39, sequence length 2048, 6260
  optimizer steps: 159,744 tokens/step and 999,997,440 tokens/run. No repetition.
- Same frozen FineWeb/tokenizer shards and the prior optimizer/schedule choices.
  Architecture parameter counts differ, so this is not a parameter-matched study.
- Validation covers **all 2048 complete blocks** (1024 batches of two), avoiding
  a seed-dependent subset. This differs from the old 1536-block validation.
  Do not pool old and new endpoint scores as replicates.
- Dense BF16/FP32-accumulate MFU denominator: 165.2 TFLOP/s. Software estimate
  only. Compilation, gradient checkpointing and dropout remain disabled.
- Keep one checkpoint; save at 3130 and 6260, reducing disk footprint. Do not
  overwrite or prune old runs. A minimum free-space check is required before
  any full run; global filesystem capacity is not proof of account quota.

Gates: local tests and frozen source/config digests; then separate deep/wide
GPU probes with three real optimizer updates, finite nonzero gradients, changed
parameters and one finite validation batch. Probe pass is engineering evidence
only. Full training requires both probes to pass and safe storage availability.
Resource ceiling for this protocol is 12 single-GPU runs, at most four concurrent
initially, each capped at 12 hours. Do not submit replacement/retry runs silently.

Primary analysis: per-seed WSD-minus-cosine endpoint validation loss within each
architecture, followed by seed-level mean and dispersion. Three seeds are still
small; do not treat validation tokens as independent experimental replicates.
No test-set-guided tuning, cherry-picking or automatic claim of downstream gains.
Downstream checkpoint evaluation is a separate fixed-protocol follow-up.
