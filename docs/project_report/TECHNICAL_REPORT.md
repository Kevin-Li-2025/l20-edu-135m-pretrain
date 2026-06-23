# L20-edu-135M Technical Report
## Current Status
- Current 10B run tokens: 786,432,000
- Recent throughput: 48,999 tok/s
- Recent MFU: 50.51%
- Estimated remaining time to 10B: 52.2 hours

## Six-Task Comparison
| Model | Mean | Random-adjusted mean | ARC-C | ARC-E | HellaSwag | LAMBADA | PIQA | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 0.4917 | 0.2661 | 0.2969 | 0.5854 | 0.4301 | 0.4289 | 0.6839 | 0.5249 |
| SmolLM-135M | 0.4767 | 0.2491 | 0.2875 | 0.5610 | 0.4265 | 0.3757 | 0.6823 | 0.5272 |
| L20-edu-135M final a0875 | 0.4150 | 0.1636 | 0.2867 | 0.4958 | 0.3240 | 0.2602 | 0.6148 | 0.5083 |
| OPT-125M | 0.4099 | 0.1580 | 0.2210 | 0.3990 | 0.3160 | 0.3856 | 0.6202 | 0.5178 |
| GPT-2 Small | 0.3954 | 0.1407 | 0.2261 | 0.3973 | 0.3138 | 0.3076 | 0.6208 | 0.5067 |
| Pythia-160M | 0.3544 | 0.0927 | 0.2312 | 0.3641 | 0.3030 | 0.1225 | 0.5979 | 0.5075 |
| Cerebras-GPT-111M | 0.3491 | 0.0861 | 0.2099 | 0.3506 | 0.2720 | 0.1912 | 0.5811 | 0.4901 |

## What Is Already Supported By Evidence
- Speed design: 2K context + Liger + compile + micro-batch 16 is the measured fastest training setup.
  Best benchmark: `sdpa+liger+compile:16` at 49,119 tok/s.
- Data quality: Stage4 metadata records cross-source near-duplicate removal, segment deduplication, 13-gram contamination checks, and LCS filtering.
- SFT design: interpolation ablation selected `stage4-sft-a0875`; full SFT was worse than the fused checkpoint.
- Efficiency: current run sustains about 50% MFU on a single L20.

## Evidence Gaps To Close
- Per-source data ablation requires additional controlled short runs; current evidence is correlational for the mixture weights.
- Long-context curriculum contribution needs checkpoints at 2K-only, 4K, and 8K evaluated under the same six-task harness.
- Data cleaning contribution needs a small no-crossdedup or relaxed-filter control, capped to avoid contamination claims.
- Statistical robustness needs at least one repeat seed or bootstrap confidence intervals for benchmark deltas.

## Generated Artifacts
- `benchmark_comparison.csv`
- `speed_ablation.csv`
- `sft_interpolation_ablation.csv`
- `training_events.csv`
- `compute_energy_summary.json`
- `data_quality_summary.json`
- `training_loss.svg`
