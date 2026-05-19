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

## Training Recipe

| Field | Value |
| --- | --- |
| Sequence length | 2048 tokens |
| Micro batch size | 6 sequences |
| Gradient accumulation | 43 steps |
| Global batch size | 258 sequences |
| Tokens / optimizer step | 528,384 |
| Max steps | 18,928 |
| Optimizer | AdamW |
| Peak learning rate | `4e-4` |
| LR schedule | linear warmup + cosine decay to `0.1 * peak_lr` |
| Warmup | 1,000 steps |
| Weight decay | 0.1 |
| Adam beta1 / beta2 | 0.9 / 0.95 |
| Gradient clipping | 1.0 |
| Precision | bfloat16 |
| Gradient checkpointing | enabled |
| Torch compile | enabled |
| Eval interval | 500 steps |
| Checkpoint interval | 1,000 steps, keeping the last 2 regular checkpoints |

The training config is included in the repository as
`configs/l20_135m_deepthin.yaml`. A fuller recipe is available in
`docs/training_recipe.md`.

## Runtime And Cost Notes

| Field | Value |
| --- | --- |
| GPU | NVIDIA L20 |
| Reported GPU memory | 46,068 MiB total |
| Mean logged throughput | 38,541 tokens/s |
| Mean logged throughput after step 1,000 | 38,587 tokens/s |
| Estimated training time | about 72 GPU-hours |
| Final checkpoint mtime | 2026-05-19 05:04:22 +0800 |

Exact peak VRAM and billing cost were not logged. A reproducible cost estimate
is `72 GPU-hours * L20_hourly_rate`, excluding storage, network egress, idle
time, and engineering time.

Known run issues:

- Transient dataset mirror read timeouts occurred near the end of training and
  recovered through retry.
- The final perplexity command printed `loss=2.8731 perplexity=17.69`, then hit
  a Python finalization crash. The metric is reported because it was printed
  before process teardown.
- The training process printed `terminate called without an active exception`
  after the final checkpoint had been written. The final checkpoint was
  load-tested with `AutoModelForCausalLM`.

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

Training artifacts:

- `training/training_metrics.csv`
- `training/training_summary.json`
- `training/loss_curve_zoom.png`
- `training/training_curves.png`

![Loss curve after warmup](training/loss_curve_zoom.png)

![Training curves](training/training_curves.png)

Zero-shot `lm-eval` results for the final checkpoint:

| Task | Metric | Score |
| --- | --- | ---: |
| ARC-Challenge | acc_norm | 0.2765 |
| ARC-Easy | acc_norm | 0.5059 |
| HellaSwag | acc_norm | 0.3272 |
| LAMBADA OpenAI | acc | 0.2540 |
| PIQA | acc_norm | 0.6224 |
| WinoGrande | acc | 0.5099 |

### Benchmark Protocol

Candidate and public baseline numbers were run with the same evaluation setup:

| Field | Setting |
| --- | --- |
| Harness | EleutherAI `lm-evaluation-harness` |
| Harness version | `0.4.12` |
| Backend | `--model hf` |
| Device | `cuda:0` |
| Dtype | `bfloat16` |
| Batch size | `auto`, resolved to 64 |
| Few-shot setting | zero-shot |
| Dataset limit | none; full task datasets |
| Samples | `--log_samples` enabled |
| Seeds | harness defaults: Python 0, NumPy 1234, Torch 1234, few-shot 1234 |
| Candidate numbers | self-run on the final checkpoint |
| Baseline numbers | self-run through `scripts/eval_public_baselines.sh`, not copied from leaderboards |

The public baselines were evaluated with the same harness version, task list,
zero-shot setting, dtype, device class, batch policy, and comparison parser. They
were **not** evaluated with the same tokenizer or model context length; each
public model used its own released tokenizer and Hugging Face model config. This
is therefore a public-model benchmark comparison, not a controlled
same-tokenizer architecture comparison.

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
- `docs/evaluation_report.md`

### Contamination Status

No full benchmark contamination pass is claimed for this release. The project
repository includes `scripts/check_contamination.py` and
`scripts/sample_training_text.py`, but a separate audit against ARC, HellaSwag,
PIQA, LAMBADA, and WinoGrande samples was not completed before release. Because
the model was trained on a public web-scale FineWeb-Edu slice, benchmark overlap
cannot be ruled out without that audit.

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
