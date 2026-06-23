# Documentation

This directory separates stable public evidence from exploratory notes.

## Core Artifacts

- [project_report/TECHNICAL_REPORT.md](project_report/TECHNICAL_REPORT.md):
  consolidated benchmark, compute, data-quality, SFT, and RLVR report.
- [evaluation_report.md](evaluation_report.md): original base-model evaluation
  protocol and public baseline comparison notes.
- [training_recipe.md](training_recipe.md): initial single-L20 pretraining
  recipe.
- [reproducibility.md](reproducibility.md): artifact map, excluded files, and
  expected reproduction path.

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

Exploratory files are kept when they explain a design decision, but the README
and technical report should be treated as the public entry points.
