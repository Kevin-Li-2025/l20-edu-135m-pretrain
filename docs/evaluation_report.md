# Evaluation Report

This report summarizes the final `l20-edu-135m` base checkpoint and the public
baseline comparison run.

## Candidate

- Model: `l20-edu-135m-deepthin`
- Released checkpoint: `runs/l20-edu-135m-deepthin/step-018928`
- Hugging Face repo: `AliceYin/l20-edu-135m`
- Parameters: 134,515,008
- Training tokens: 10,001,252,352 planned tokens
- Tokens per parameter: 74.35
- Dataset: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`
- Final validation: loss `2.8731`, perplexity `17.69`

## Training Curves

The training log contains 1,903 train-loss points and 38 validation-loss points.
The extracted metrics are committed as `docs/training_metrics.csv`, with a small
machine-readable summary in `docs/training_summary.json`.

![Loss curve after warmup](assets/loss_curve_zoom.png)

![Training curves](assets/training_curves.png)

## lm-eval Results

The final checkpoint was evaluated with EleutherAI `lm-evaluation-harness` on a
small-model base-LM suite. All public baselines were run through the same task
set and comparison script.

| Task | Metric | l20-edu-135m | gpt2-small | opt-125m | gpt-neo-125m | cerebras-gpt-111m | pythia-160m | smollm-135m | smollm2-135m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.2765 | 0.2261 | 0.2210 | 0.2321 | 0.2099 | 0.2312 | 0.2875 | 0.2969 |
| ARC-Easy | acc_norm | 0.5059 | 0.3973 | 0.3990 | 0.3965 | 0.3506 | 0.3641 | 0.5610 | 0.5854 |
| HellaSwag | acc_norm | 0.3272 | 0.3138 | 0.3160 | 0.3055 | 0.2720 | 0.3030 | 0.4265 | 0.4301 |
| LAMBADA OpenAI | acc | 0.2540 | 0.3076 | 0.3856 | 0.3765 | 0.1912 | 0.1225 | 0.3757 | 0.4289 |
| PIQA | acc_norm | 0.6224 | 0.6208 | 0.6202 | 0.6213 | 0.5811 | 0.5979 | 0.6823 | 0.6839 |
| WinoGrande | acc | 0.5099 | 0.5067 | 0.5178 | 0.5099 | 0.4901 | 0.5075 | 0.5272 | 0.5249 |

Win rates over public baselines:

| Baseline | Wins / Tasks | Win Rate |
| --- | ---: | ---: |
| GPT-2 small | 5 / 6 | 0.833 |
| OPT-125M | 4 / 6 | 0.667 |
| GPT-Neo-125M | 4 / 6 | 0.667 |
| Cerebras-GPT-111M | 6 / 6 | 1.000 |
| Pythia-160M | 6 / 6 | 1.000 |
| SmolLM-135M | 0 / 6 | 0.000 |
| SmolLM2-135M | 0 / 6 | 0.000 |

## Training Budget Context

The comparison is useful, but it is not a matched training-budget comparison.
The public baselines were trained with very different corpora and token budgets.

| Model | Parameters | Reported Training Data / Tokens | Notes |
| --- | ---: | --- | --- |
| l20-edu-135m | 134.5M | 10B FineWeb-Edu tokens | This project, single L20 run |
| GPT-2 small | 124M | WebText, about 40GB text from 8M documents | Official GPT-2 reporting does not give a clean token count |
| OPT-125M | 125M | 180B tokens | OPT model family training budget |
| GPT-Neo-125M | 125M | The Pile, commonly reported as 300B tokens | Public GPT-Neo checkpoint |
| Cerebras-GPT-111M | 111M | About 2.2B tokens | 20 tokens per parameter recipe |
| Pythia-160M | 160M | About 300B tokens | Pythia suite training budget |
| SmolLM-135M | 135M | 600B tokens | SmolLM-Corpus |
| SmolLM2-135M | 135M | 2T tokens | FineWeb-Edu, DCLM, The Stack, and curated data |

The strongest honest reading is:

> `l20-edu-135m` is competitive with several older 100M-160M public base models
> while using only 10B pretraining tokens on one L20. It is clearly behind modern
> compact models such as SmolLM and SmolLM2, which use much larger data budgets.

The model should not be described as SOTA. A controlled architecture claim still
requires training `configs/l20_wide_140m_baseline.yaml` under the same tokenizer,
data, optimizer, schedule, and token budget.

## Interpretation

The model learned usable base-LM behavior: it can continue text, complete simple
facts sometimes, and score above several older baselines on commonsense and
reading-style multiple-choice tasks.

The main weaknesses are expected for a 135M base model trained on 10B tokens:

- weak instruction following
- unstable factual recall
- repetition during free-form generation
- lower LAMBADA accuracy than GPT-2/OPT/GPT-Neo and modern SmolLM models
- no chat alignment or safety tuning

For public presentation, frame this as a training systems and reproducibility
project rather than as a high-quality assistant model.

## Reproducibility Artifacts

- Training config: `configs/l20_135m_deepthin.yaml`
- Eval comparison script: `scripts/compare_lm_eval.py`
- Public baseline runner: `scripts/eval_public_baselines.sh`
- Raw comparison files on Hugging Face: `eval/comparison.md`,
  `eval/comparison.json`
- Local ignored artifacts after a run: `eval_results/`, `logs/`, `runs/`

## Sources For Baseline Budget Notes

- GPT-2 / WebText: https://openai.com/index/better-language-models/
- OPT model family: https://arxiv.org/abs/2205.01068
- GPT-Neo-125M: https://huggingface.co/EleutherAI/gpt-neo-125m
- Cerebras-GPT-111M: https://huggingface.co/cerebras/Cerebras-GPT-111M
- Pythia suite: https://github.com/EleutherAI/pythia
- SmolLM: https://huggingface.co/blog/smollm
- SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
