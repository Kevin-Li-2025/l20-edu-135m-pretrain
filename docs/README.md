# Documentation

This directory separates stable public evidence from exploratory notes.

## Core Artifacts

- [project_report/TECHNICAL_REPORT.md](project_report/TECHNICAL_REPORT.md):
  consolidated benchmark, compute, data-quality, SFT, and RLVR report.
- [project_report/ablation_plan.json](project_report/ablation_plan.json):
  machine-checkable plan for the next controlled experiments.
- [evaluation_report.md](evaluation_report.md): original base-model evaluation
  protocol and public baseline comparison notes.
- [training_recipe.md](training_recipe.md): initial single-L20 pretraining
  recipe.
- [reproducibility.md](reproducibility.md): artifact map, excluded files, and
  expected reproduction path.
- [A40_MAX_EFFICIENCY_RESEARCH.md](A40_MAX_EFFICIENCY_RESEARCH.md): measured
  five-A40 continuation design, throughput experiments, and quality gates.
- [../A40_RUNBOOK.md](../A40_RUNBOOK.md): deployment and topology checks for
  five- or six-A40 nodes.

## Figures And Structured Summaries

- [assets/](assets/): loss and training-curve images for the model card and
  paper.
- [project_report/benchmark_comparison.csv](project_report/benchmark_comparison.csv):
  compact six-task comparison table.
- [project_report/compute_energy_summary.json](project_report/compute_energy_summary.json):
  compute and energy accounting.
- [project_report/data_quality_summary.json](project_report/data_quality_summary.json):
  Stage 4 filtering summary.
- [project_report/sft_interpolation_ablation.csv](project_report/sft_interpolation_ablation.csv):
  SFT interpolation sweep.
- [project_report/speed_ablation.csv](project_report/speed_ablation.csv):
  measured throughput candidates.
- Validate the ablation plan with:

```bash
python scripts/check_ablation_plan.py
```

## Data Cleaning Tools

- `scripts/prepare_skill_targeted_corpus.py` normalizes, quality-filters,
  skill-tags, deduplicates, and benchmark-screens targeted curriculum data.
- `scripts/eval_and_reweight_mixture.py` converts six-task evaluation gaps into
  next-stage skill mixture weights.
- `scripts/audit_l20_mfu_config.py` checks whether a training config is likely
  to underuse the L20 before starting an expensive run.

Exploratory files are kept when they explain a design decision, but the README
and technical report should be treated as the public entry points.
