# SFT Recipe

This document defines the supervised fine-tuning version of the base checkpoint.
It is a recipe and implementation scaffold; it should not be presented as a
released chat model until the SFT run and post-training evals have completed.

## Target Model

| Field | Value |
| --- | --- |
| SFT name | `l20-edu-135m-sft` |
| Base model | `AliceYin/l20-edu-135m` |
| Output directory | `runs/l20-edu-135m-sft` |
| Training script | `python -m l20_pretrain.train_sft` |
| Config | `configs/l20_edu_135m_sft.yaml` |
| Prompt format | `### System`, `### Instruction`, optional `### Input`, `### Response` |
| Loss mask | assistant response only by default |

## Default Data

The default config uses `HuggingFaceH4/ultrachat_200k`:

| Field | Value |
| --- | --- |
| Dataset config | `default` |
| Train split | `train_sft` |
| Eval split | `test_sft` |
| Max train examples | 50,000 |
| Max eval examples | 1,024 |
| Max raw example size | 12,000 characters |
| Context length | 2048 |

The loader also supports local JSONL files. Each row can use either
instruction-style columns:

```jsonl
{"instruction":"Explain attention in one paragraph.","input":"","output":"Attention lets a model compare tokens..."}
{"prompt":"The capital of New Zealand is","response":" Wellington."}
```

or chat-style messages:

```jsonl
{"messages":[{"role":"user","content":"Write a haiku about GPUs."},{"role":"assistant","content":"Silent tensor cores..."}]}
```

For chat rows, the final assistant message is the supervised target. Earlier
messages are context.

## Optimization

| Field | Value |
| --- | --- |
| Micro batch size | 8 sequences |
| Gradient accumulation | 8 |
| Global batch size | 64 sequences |
| Max steps | 1,200 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| LR schedule | linear warmup + cosine decay |
| Warmup steps | 100 |
| Weight decay | 0.0 |
| Precision | bfloat16 |
| Gradient checkpointing | enabled |
| Eval interval | 100 steps |
| Save interval | 200 steps |

This is intentionally conservative for a 135M base model. If eval loss rises or
free-form generations collapse into repetitive chat templates, reduce the
learning rate to `1e-5` or lower `max_steps`.

## Run

From the repository root:

```bash
python -m l20_pretrain.train_sft configs/l20_edu_135m_sft.yaml
```

Resume:

```bash
python -m l20_pretrain.train_sft configs/l20_edu_135m_sft.yaml \
  --resume runs/l20-edu-135m-sft/step-000800
```

Use a local JSONL file:

```bash
python -m l20_pretrain.train_sft configs/l20_edu_135m_sft.yaml
```

with this config override:

```yaml
dataset:
  local_jsonl_path: data/my_sft.jsonl
  name:
  split: train
  eval_split:
```

## Expected Behavior After SFT

A good SFT run should improve:

- following simple instructions
- direct short-answer QA
- structured answer formatting
- short writing prompts
- refusal to continue arbitrary pretraining-style text

It will not make the model broadly factual or reasoning-strong. The base model is
only 135M parameters trained on 10B tokens, so SFT mainly teaches interface and
style. It cannot reliably create knowledge that the base model did not learn.

## Eval Checklist

Before publishing `l20-edu-135m-sft`, run:

- base completion sanity prompts
- instruction-following prompts
- factual QA prompts, including capitals and simple geography
- short writing prompts
- JSON formatting prompts
- repetition checks at temperature 0.7 and 0.9
- `lm-eval` on the original base-model suite to measure regression
- a small held-out SFT eval loss pass

Keep the base model and SFT model separate on Hugging Face. The SFT model card
should say clearly that it is instruction-tuned and should include the SFT data
source, example count, optimizer, training time, and eval results.
