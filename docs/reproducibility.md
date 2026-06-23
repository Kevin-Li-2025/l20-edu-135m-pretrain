# Reproducibility Manifest

This project aims to make the small-model training record inspectable without
committing large or private artifacts.

## Public Artifacts

- Model weights and tokenizer: `AliceYin/l20-edu-135m` on Hugging Face.
- Training and evaluation code: `src/l20_pretrain/` and `scripts/`.
- Training configs: `configs/`.
- Curated benchmark summaries: `results/`.
- Consolidated report: `docs/project_report/TECHNICAL_REPORT.md`.
- Paper draft: `paper/l20_edu_135m_arxiv.pdf`.

## Excluded Artifacts

The following are intentionally not tracked:

- checkpoints and optimizer states;
- raw datasets and packed token shards;
- raw `lm-eval` sample outputs;
- run logs and process logs;
- local virtual environments and CUDA package caches;
- Hugging Face tokens, SSH keys, and service credentials.

The `.gitignore` and `scripts/check_repo_hygiene.py` enforce this boundary.

## Reproduction Levels

Level 1 checks the repository itself:

```bash
python scripts/check_repo_hygiene.py
python -m pytest -q
```

Level 2 reproduces evaluation from a downloaded checkpoint:

```bash
scripts/setup_eval_env.sh
scripts/eval_lm_harness.sh /path/to/checkpoint
```

Level 3 reruns training. This requires a CUDA host, a compatible PyTorch build,
dataset access, and enough local disk for packed shards and checkpoints:

```bash
python -m l20_pretrain.train configs/l20_135m_deepthin.yaml
python -m l20_pretrain.train configs/l20_edu_135m_stage4_hq_crossdedup_8k.yaml
```

Exact wall-clock time depends on host I/O, PyTorch/CUDA versions, context
length, and whether the optimized short-context configuration is used.

## Benchmark Caveats

The six-task public comparison uses the same harness protocol where practical,
but public baselines keep their released tokenizers and architecture settings.
The comparison supports a practical quality and token-budget discussion, not a
strict architecture-isolated claim.

The contamination gate is documented and committed, but no absolute
contamination-free claim is made.
