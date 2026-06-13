# Ultra CPT + Anti-Forgetting SFT Plan

## Research Conclusions

The next run should optimize for data quality and evaluation feedback, not only
raw token count.

Primary sources used:

- DataComp-LM shows that model-based filtering is a key lever for high-quality
  web pretraining data, and that DCLM-style curation can improve compute
  efficiency.
- FineWeb/FineWeb-Edu documents large-scale Common Crawl cleaning, educational
  classifier filtering, and the importance of deduplication and filtering
  ablations.
- SmolLM2 reports that FineWeb-Edu plus DCLM is the strongest open small-model
  English web backbone; DCLM-Edu is explicitly intended for small models such
  as 135M and 360M.
- Tulu 3 shows that modern post-training benefits from skill-specific data
  mixtures, decontamination, iterative evaluation, and selective addition or
  removal of datasets.
- LIMA supports the principle that instruction tuning should prioritize a small
  amount of high-quality diverse data because most knowledge should come from
  pretraining.
- Forgetting work on finetuning shows that mixing generic/replay data during
  finetuning reduces drift relative to narrow-domain SFT.
- Liger Kernel and PyTorch performance guidance support fused kernels and larger
  effective micro-batches where memory allows.

## CPT Recipe

Stage 3 is a 10B-token single-GPU continual-pretraining run:

- 40% DCLM-Edu, filtered with `edu_score >= 3` / `edu_int_score >= 3`
- 35% SmolLM FineWeb-Edu dedup, filtered with score/int-score >= 4
- 10% replay from the prior educational long-context corpus
- 4% Cosmopedia v2
- 6% FineMath 4+
- 5% Stack-Edu code

The large DCLM/FineWeb-Edu backbone targets the current gap to SmolLM on
HellaSwag, PIQA, ARC-Easy, and LAMBADA. Replay limits catastrophic drift from
the existing model; math/code tails preserve Stage 2 improvements.

## SFT Recipe

After CPT, run a short low-learning-rate SFT:

- Tulu 3 SFT mixture as the instruction-following backbone
- 15% LM-continuation replay decoded from the CPT tokenized shard
- balanced selection across reasoning, factual QA, writing, formatting, coding,
  safety, and general categories
- response quality filtering, deduplication, and length/repetition filtering

The goal is useful instruction following without overwriting the base model's
general language-modeling distribution.

## Evaluation Gate

No run can guarantee SOTA before it is trained and evaluated. The release gate is
therefore empirical:

- run the same six-task SmolLM target suite:
  `arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,winogrande`
- compare against `HuggingFaceTB/SmolLM-135M` and `HuggingFaceTB/SmolLM2-135M`
- track per-task gaps and mean score
- only call a checkpoint a release candidate if it improves the mean and does
  not regress key skills unexpectedly

## Efficiency Plan

- keep Liger Kernel enabled
- use bfloat16 and SDPA on the current CUDA stack
- increase Stage 3 micro-batch from 3 to 4 and reduce accumulation from 22 to 16
  to reduce accumulation overhead while keeping tokens per optimizer step close
  to the previous setup
- keep a safe fallback config with micro-batch 3 if the fast config OOMs
- monitor `tokens_per_sec_window`, `step_time_sec_window`, and `mfu_pct`
