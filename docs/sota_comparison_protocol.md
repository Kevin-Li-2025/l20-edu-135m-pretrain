# SOTA and Comparison Protocol

## Bottom line

Do not claim general SOTA for this project.

For a single L20 run, the credible claim is narrower:

> Best controlled 100M-150M base model in our experiment under the same
> tokenizer, FineWeb-Edu slice, filtering, context length, optimizer family, and
> token budget.

If the model beats public 100M-200M base models on a fixed public eval suite, the
stronger claim is:

> Competitive with or better than public 100M-200M open base models on reported
> zero/few-shot benchmarks, with documented training data and contamination
> checks.

## Preconditions for a serious comparison

1. Define the lane.

   Compare base model to base model, not base to instruct/chat. Keep parameter
   class explicit: for this project, `100M-150M` or `100M-200M`. Report whether
   the comparison is controlled, public-benchmark, or leaderboard-style.

2. Fix the controlled variables.

   If the claim is architecture quality, keep tokenizer, data source, data
   filter, context length, token budget, optimizer, LR schedule, precision, and
   checkpoint selection rule identical. The only intended variable should be the
   architecture.

3. Compare to the right baselines.

   Minimum public baselines:

   - GPT-2 small, 124M
   - OPT-125M
   - GPT-Neo-125M
   - Cerebras-GPT-111M
   - Pythia-160M
   - SmolLM-135M
   - SmolLM2-135M

   Minimum controlled baselines in this repo:

   - `configs/l20_wide_140m_baseline.yaml`
   - `configs/l20_135m_deepthin.yaml`

4. Use a standard evaluation harness.

   Use EleutherAI `lm-evaluation-harness` for academic zero/few-shot tasks and
   save raw results plus samples. Hugging Face Open LLM Leaderboard uses this
   family of tooling and publishes task details; their v2-style tasks include
   IFEval, BBH, MATH Level 5, GPQA, MuSR, and MMLU-Pro.

5. Evaluate base-model fit separately from downstream QA.

   For a pretrained base model, perplexity still matters, but one held-out
   corpus is not enough. Paloma is relevant because it reports perplexity across
   hundreds of English and code domains instead of assuming one validation set
   generalizes.

6. Control checkpoint selection.

   Decide before training which checkpoint is final. Do not scan many
   checkpoints on the benchmark suite and report the best one unless that
   selection rule is also applied to every baseline.

7. Run contamination checks.

   At minimum, document whether benchmark prompts/answers were included in the
   training corpus, run n-gram overlap checks where feasible, and treat suspicious
   outlier scores as invalid until investigated. Recent contamination work shows
   that contamination is hard to define precisely and can inflate benchmark
   scores.

8. Report uncertainty and raw artifacts.

   Keep raw JSON result files, model generations, harness version, command line,
   batch size, dtype, commit hash, GPU, seed, and checkpoint path. Small deltas
   are not meaningful if they fall within harness variance or are caused by
   prompt/template differences.

9. Publish a usable model card.

   Include intended use, limitations, model architecture, tokenizer, training
   data, filtering, token budget, optimizer, eval tasks, metrics, contamination
   statement, compute, and license.

## Evaluation tiers for this project

### Tier A: Controlled architecture win

Goal: prove the deep-thin architecture beats a wide baseline under identical
training conditions.

Required:

- train `configs/l20_135m_deepthin.yaml`
- train `configs/l20_wide_140m_baseline.yaml`
- same token budget, tokenizer, data, filtering, and evaluation
- compare validation loss, perplexity, and lm-eval suite

This is the cleanest claim we can make from one L20.

### Tier B: Public 120M-class comparison

Goal: compare against existing public models.

Required:

- evaluate our checkpoint and all public baselines with the same command
- use identical task list, dtype, batch policy, and few-shot setting
- report model size, training tokens if known, tokenizer, and data notes

Suggested tasks:

- `lambada_openai`
- `hellaswag`
- `piqa`
- `arc_easy`
- `arc_challenge`
- `winogrande`

These are more informative for 100M-class base models than very hard modern
leaderboard tasks where many small models will sit near chance.

### Tier C: Leaderboard-style claim

Goal: make a public leaderboard-compatible claim.

Required:

- run the exact leaderboard task group and harness version
- disclose whether the model is base, continued-pretrained, fine-tuned, or chat
- publish raw results and samples
- submit or reproduce against the same leaderboard backend

This is useful for visibility, but it is not the first proof point for a 135M
base model.

## Recommended commands

Install the harness separately:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness /tmp/lm-evaluation-harness
cd /tmp/lm-evaluation-harness
pip install -e ".[hf]"
```

Run our base-model suite:

```bash
scripts/eval_lm_harness.sh runs/l20-edu-135m-deepthin/step-010000
```

Run the same suite on a public baseline:

```bash
scripts/eval_lm_harness.sh EleutherAI/pythia-160m eval_results/pythia-160m
```

Run a leaderboard-style eval:

```bash
scripts/eval_openllm_leaderboard.sh runs/l20-edu-135m-deepthin/step-010000
```

## Sources

- Hugging Face leaderboard docs: https://huggingface.co/docs/leaderboards/main/index
- Open LLM Leaderboard methodology: https://huggingface.co/docs/leaderboards/main/open_llm_leaderboard/about
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- HELM paper page: https://huggingface.co/papers/2211.09110
- Paloma benchmark: https://papers.nips.cc/paper_files/paper/2024/hash/760b2d94398aa61468aa3bc11506d9ea-Abstract-Datasets_and_Benchmarks_Track.html
- Pythia paper: https://arxiv.org/abs/2304.01373
- LLM360 transparency post: https://www.llm360.ai/news/introducing-llm360-fully-transparent-open-source-llms.html
- Benchmark contamination survey: https://arxiv.org/abs/2406.04244
- Evaluation contamination measurement: https://arxiv.org/abs/2411.03923
- ConStat contamination paper: https://arxiv.org/abs/2405.16281
- Model cards docs: https://huggingface.co/docs/hub/main/en/model-cards
