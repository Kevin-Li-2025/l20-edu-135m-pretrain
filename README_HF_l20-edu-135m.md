---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- causal-lm
- pretraining
- from-scratch
- fineweb-edu
- single-gpu
- l20
datasets:
- HuggingFaceFW/fineweb-edu
---

# l20-edu-135m

`l20-edu-135m` is a 134.5M-parameter Llama-style causal language model
pretrained from scratch on 10B FineWeb-Edu tokens using a single NVIDIA L20 GPU.

This is a **base model**, not an instruction-tuned chat model. It is released as
a small-model pretraining artifact for research, evaluation, reproducibility,
continued pretraining, and downstream supervised fine-tuning.

## Model Details

| Field | Value |
| --- | --- |
| Model type | Decoder-only causal LM |
| Architecture | Llama-style Transformer |
| Parameters | 134,515,008 |
| Layers | 30 |
| Hidden size | 576 |
| FFN size | 1536 |
| Attention | 9 query heads, 3 key/value heads |
| Context length | 2048 |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` tokenizer |
| Training data | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |
| Training budget | 10,001,252,352 planned tokens |
| Tokens / parameter | 74.35 |
| Final checkpoint | step 18,928 |
| Hardware | single NVIDIA L20 GPU |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "AliceYin/l20-edu-135m"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Because this is a base model, completion-style prompts work better than
instruction/chat prompts.

## Evaluation

Final validation:

- Loss: `2.8731`
- Perplexity: `17.69`

Zero-shot `lm-eval` results for the final checkpoint:

| Task | Metric | Score |
| --- | --- | ---: |
| ARC-Challenge | acc_norm | 0.2765 |
| ARC-Easy | acc_norm | 0.5059 |
| HellaSwag | acc_norm | 0.3272 |
| LAMBADA OpenAI | acc | 0.2540 |
| PIQA | acc_norm | 0.6224 |
| WinoGrande | acc | 0.5099 |

Public baseline win rates on the same task set:

| Baseline | Wins / Tasks | Win Rate |
| --- | ---: | ---: |
| GPT-2 small | 5 / 6 | 0.833 |
| OPT-125M | 4 / 6 | 0.667 |
| GPT-Neo-125M | 4 / 6 | 0.667 |
| Cerebras-GPT-111M | 6 / 6 | 1.000 |
| Pythia-160M | 6 / 6 | 1.000 |
| SmolLM-135M | 0 / 6 | 0.000 |
| SmolLM2-135M | 0 / 6 | 0.000 |

The full comparison artifacts are included in this repository under:

- `eval/comparison.md`
- `eval/comparison.json`

## Interpretation

This model is competitive with several older 100M-160M public base models on a
matched `lm-eval` task suite while using only 10B pretraining tokens on one L20.
It is not SOTA and does not beat modern heavily overtrained compact models such
as SmolLM-135M or SmolLM2-135M, which use substantially larger token budgets.

## Intended Use

This checkpoint is suitable for:

- base model evaluation
- continued pretraining experiments
- supervised fine-tuning experiments
- small-model training pipeline demonstrations
- studying single-GPU pretraining tradeoffs

It is not suitable as a production assistant without post-training, safety
evaluation, and domain-specific validation.

## Limitations

- This is a small base model trained on 10B tokens.
- It is not instruction-tuned and may not follow user requests reliably.
- It can produce incorrect facts, repetition, or incomplete generations.
- It has not been safety aligned.
- Benchmark results should not be interpreted as general assistant quality.
- Results should not be described as SOTA without controlled matched-budget
  baselines and contamination checks.

## Training-Budget Context

For fair interpretation, training data size matters:

| Model | Reported Training Budget |
| --- | --- |
| `l20-edu-135m` | 10B FineWeb-Edu tokens |
| GPT-2 small | WebText, about 40GB text; no clean official token count |
| OPT-125M | 180B tokens |
| GPT-Neo-125M | The Pile, commonly reported as 300B tokens |
| Cerebras-GPT-111M | about 2.2B tokens |
| Pythia-160M | about 300B tokens |
| SmolLM-135M | 600B tokens |
| SmolLM2-135M | 2T tokens |

## Citation

If you use this checkpoint, please cite or link to this repository and include
the training-token budget when comparing against other compact language models.
