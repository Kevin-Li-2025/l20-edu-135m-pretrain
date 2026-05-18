# 120M-Class Architecture Search

## Public reference models

| Model | Approx size | Shape | Tokenizer / context | Training data notes |
| --- | ---: | --- | --- | --- |
| GPT-2 small | 124M | 12 layers, 768 hidden, 12 heads | GPT-2 BPE, 1024 context | WebText-style corpus. |
| OPT-125M | 125M | 12 layers, 768 hidden, 12 heads | GPT-2-style BPE, 2048 context | Meta OPT corpus, public model card says 180B training tokens for OPT family. |
| GPT-Neo-125M | 125M | 12 layers, 768 hidden, 12 heads | GPT-2 BPE, 2048 context | The Pile. |
| Cerebras-GPT-111M | 111M | 10 layers, 768 hidden, 12 heads | GPT-2 BPE, 2048 context | The Pile, compute-optimal scaling suite. |
| Pythia-160M | 160M | 12 layers, 768 hidden, 12 heads | GPT-NeoX tokenizer, 2048 context | The Pile, 300B-token training suite. |
| SmolLM-135M | 135M | 30 layers, 576 hidden, 9 Q heads, 3 KV heads | SmolLM tokenizer, 2048 context | 600B-token mixture with FineWeb-Edu, Cosmopedia v2, Python-Edu. |
| SmolLM2-135M | 135M | 30 layers, 576 hidden, 9 Q heads, 3 KV heads | SmolLM2 tokenizer, 8192 context | 2T-token mixture with FineWeb-Edu, DCLM, The Stack, and filtered data. |
| MobileLLM-125M | 125M | Deep-thin Llama-like design | Gated model weights | Paper argues deep-thin, embedding sharing, and GQA beat older wide 125M designs. |

## Answer

Yes, but only under a precise baseline.

We have a realistic chance to beat old 120M-class wide models, such as GPT-2
small, OPT-125M, GPT-Neo-125M, and Cerebras-GPT-111M, when we control tokenizer,
data, and token count. Their shape is mostly 10-12 layers at 768 hidden size.
The stronger modern recipe is deep-thin: more layers, smaller hidden size,
SwiGLU, RMSNorm, RoPE, grouped-query attention, tied embeddings, and clean token
packing.

Beating SmolLM2-135M by architecture alone is much harder. SmolLM2 already uses
the same broad deep-thin recipe and was trained for roughly 2T tokens. On one
L20, the practical target is a controlled 10B-100B token run that beats wide
baselines at the same budget, then scale the winning architecture.

## Experiment contract

Use the same:

- tokenizer: `HuggingFaceTB/SmolLM2-135M`
- dataset: `HuggingFaceFW/fineweb-edu`, starting with `sample-10BT`
- filter: `score >= 3.0`, `int_score >= 3`, 300-50,000 chars
- context: 2048
- planned tokens: about 10B
- optimizer: AdamW beta1 0.9, beta2 0.95, cosine decay, 10% min LR

Compare:

- `configs/l20_wide_140m_baseline.yaml`: 12-layer wide model, 141.6M params.
- `configs/l20_135m_deepthin.yaml`: 30-layer deep-thin model, about 135M params.

Primary metric:

- validation loss/perplexity on held-out FineWeb-Edu stream.

Secondary metric:

- lm-eval tasks after checkpoints: ARC-Easy, HellaSwag, PIQA, WinoGrande, and
  commonsense QA if runtime permits.

## Why the deep-thin config should be the first serious run

At this parameter scale, embeddings consume a large share of the model. Tying
input and output embeddings saves tens of millions of parameters that can be
spent on depth. More layers give more serial transformations for the same token
budget, while GQA keeps KV-cache and attention projection cost manageable.

The resulting config is close to the proven 135M-class shape:

- 30 decoder layers
- 576 hidden size
- 1536 SwiGLU intermediate size
- 9 query heads
- 3 KV heads
- tied embeddings
- RoPE + RMSNorm

## Sources

- GPT-2 model card: https://huggingface.co/openai-community/gpt2
- OPT-125M model card: https://huggingface.co/facebook/opt-125m
- GPT-Neo-125M model card: https://huggingface.co/EleutherAI/gpt-neo-125m
- Cerebras-GPT-111M model card: https://huggingface.co/cerebras/Cerebras-GPT-111M
- Pythia-160M model card: https://huggingface.co/EleutherAI/pythia-160m
- SmolLM-135M model card: https://huggingface.co/HuggingFaceTB/SmolLM-135M
- SmolLM2-135M model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- MobileLLM paper: https://arxiv.org/abs/2402.14905
