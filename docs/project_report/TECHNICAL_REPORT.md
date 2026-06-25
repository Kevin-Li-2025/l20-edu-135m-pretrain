# L20-Edu-135M Technical Report

## Final Release Status

- Model: `L20-Edu-135M`
- Parameters: 134,515,008
- Hardware: single NVIDIA L20
- Public checkpoint: `AliceYin/l20-edu-135m`
- Initial pretraining: about 10B FineWeb-Edu tokens
- Stage 4 continuation: 3,000,000,965 curated tokens
- Selected release variant: `stage4-sft-a0875`
- Selected SFT interpolation: 87.5% anti-forgetting SFT checkpoint and 12.5% Stage 4 base checkpoint
- Final six-task mean: 0.4150

This report is a final-release evidence index, not a live training status page.
Historical run logs remain useful for audit, but the public claim is anchored on
the selected Stage 4/SFT checkpoint and the committed result summaries under
`results/`.

## Six-Task Comparison

| Model | Mean | Random-adjusted mean | ARC-C | ARC-E | HellaSwag | LAMBADA | PIQA | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 0.4917 | 0.2661 | 0.2969 | 0.5854 | 0.4301 | 0.4289 | 0.6839 | 0.5249 |
| SmolLM-135M | 0.4767 | 0.2491 | 0.2875 | 0.5610 | 0.4265 | 0.3757 | 0.6823 | 0.5272 |
| L20-Edu-135M final a0875 | 0.4150 | 0.1636 | 0.2867 | 0.4958 | 0.3240 | 0.2602 | 0.6148 | 0.5083 |
| OPT-125M | 0.4099 | 0.1580 | 0.2210 | 0.3990 | 0.3160 | 0.3856 | 0.6202 | 0.5178 |
| GPT-2 Small | 0.3954 | 0.1407 | 0.2261 | 0.3973 | 0.3138 | 0.3076 | 0.6208 | 0.5067 |
| Pythia-160M | 0.3544 | 0.0927 | 0.2312 | 0.3641 | 0.3030 | 0.1225 | 0.5979 | 0.5075 |
| Cerebras-GPT-111M | 0.3491 | 0.0861 | 0.2099 | 0.3506 | 0.2720 | 0.1912 | 0.5811 | 0.4901 |

## Token-Budget Framing

L20-Edu-135M is not presented as state of the art. The release is framed as a
data-efficiency and auditability result:

- compared with SmolLM-135M, the model reaches about 87.1% of the six-task mean
  using about 2.17% of the reported token budget;
- compared with SmolLM2-135M, the model reaches about 84.4% of the six-task mean
  using about 0.65% of the reported token budget;
- the comparison is useful for practical data efficiency, but it is not an
  architecture-isolated claim because public baselines use their own tokenizers,
  context settings, and pretraining recipes.

## Evidence Already Supported

- Speed design: 2K context with SDPA, Liger, compile, and micro-batch 16 is the
  measured fastest short-context setup. Best benchmark: `sdpa+liger+compile:16`
  at 49,119 tok/s.
- Data quality: Stage 4 metadata records cross-source near-duplicate removal,
  segment deduplication, 13-gram benchmark contamination checks, and LCS overlap
  filtering.
- SFT design: interpolation ablation selected `stage4-sft-a0875`; full SFT was
  worse than the fused checkpoint on the six-task suite.
- RLVR finding: GSM8K RLVR at 135M was negative; the result is retained as a
  useful lower-scale failure mode rather than converted into a success claim.
- Release hygiene: model card, paper draft, curated summaries, and repo hygiene
  checks are committed.

## Evidence Gaps To Close

The strongest next work is controlled evidence, not another blind continuation
run:

- Per-source data ablation: current mixture evidence is correlational.
- Cleaning ablation: no-crossdedup or relaxed-filter controls are needed to
  quantify the value of the Stage 4 gate.
- Context curriculum ablation: 2K-only, 2K-to-4K, and 2K-to-4K-to-8K variants
  should be compared under the same six-task harness.
- Statistical robustness: add bootstrap intervals for benchmark deltas and at
  least one repeat seed for the cheapest high-value experiment.
- MFU accounting: report pretraining MFU separately from SFT estimated MFU and
  short benchmark throughput.

## Next Experimental Plan

The planned ablations are tracked in `docs/project_report/ablation_plan.json`.
Each experiment must define:

- a single hypothesis;
- a base checkpoint;
- changed variables;
- token budget;
- expected wall-clock cost;
- required benchmark gates;
- expected artifact paths;
- stop criteria.

This keeps the project from accumulating unreviewable ad-hoc runs.

## Generated Artifacts

- `benchmark_comparison.csv`
- `speed_ablation.csv`
- `sft_interpolation_ablation.csv`
- `training_events.csv`
- `compute_energy_summary.json`
- `data_quality_summary.json`
- `training_loss.svg`
- `ablation_plan.json`
