---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- causal-lm
- supervised-fine-tuning
- instruction-tuned
- fineweb-edu
- l20
base_model:
- AliceYin/l20-edu-135m
datasets:
- HuggingFaceH4/ultrachat_200k
---

# l20-edu-135m-sft

`l20-edu-135m-sft` is the supervised fine-tuned version of
[`AliceYin/l20-edu-135m`](https://huggingface.co/AliceYin/l20-edu-135m), a
134.5M-parameter Llama-style base language model pretrained from scratch on 10B
FineWeb-Edu tokens using a single NVIDIA L20 GPU.

This model is intended as a small instruction-following checkpoint for research,
demo, and post-training experiments. It should not be treated as a production
assistant without additional safety evaluation and domain validation.

## Model Details

| Field | Value |
| --- | --- |
| Model type | Decoder-only causal LM |
| Base model | `AliceYin/l20-edu-135m` |
| Parameters | 134,515,008 |
| Context length | 2048 |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` tokenizer |
| SFT dataset | `HuggingFaceH4/ultrachat_200k` by default |
| SFT examples | fill after run |
| Final SFT checkpoint | fill after run |
| Hardware | single NVIDIA L20 GPU |

## Usage

This checkpoint uses a simple completion-style instruction format:

```text
### System:
You are a helpful, concise assistant.

### Instruction:
Write a short explanation of attention.

### Response:
```

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "AliceYin/l20-edu-135m-sft"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = """### System:
You are a helpful, concise assistant.

### Instruction:
What is the capital of New Zealand?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## SFT Recipe

| Field | Value |
| --- | --- |
| Training script | `python -m l20_pretrain.train_sft` |
| Config | `configs/l20_edu_135m_sft.yaml` |
| Sequence length | 2048 |
| Micro batch size | 8 |
| Gradient accumulation | 8 |
| Global batch size | 64 sequences |
| Max steps | 1,200 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Schedule | warmup + cosine decay |
| Warmup steps | 100 |
| Weight decay | 0.0 |
| Precision | bfloat16 |
| Loss mask | assistant response tokens only |

## Evaluation

Fill after training:

| Check | Result |
| --- | --- |
| Held-out SFT loss | TBD |
| Instruction-following sanity set | TBD |
| Factual QA sanity set | TBD |
| Short writing sanity set | TBD |
| JSON formatting sanity set | TBD |
| Base `lm-eval` regression suite | TBD |

## Limitations

- This is a 135M-parameter model, so factual recall and reasoning remain limited.
- SFT improves interface behavior, but it does not add reliable world knowledge.
- The model can still hallucinate, repeat, or fail to follow instructions.
- The model has not undergone RLHF, DPO, safety tuning, or red-team evaluation.
- Do not compare this checkpoint to larger instruction models without reporting
  model size, pretraining tokens, SFT data, and eval protocol.

## Citation

If you use this checkpoint, cite or link both the SFT model and its base model,
and report the base pretraining budget of 10B FineWeb-Edu tokens.
