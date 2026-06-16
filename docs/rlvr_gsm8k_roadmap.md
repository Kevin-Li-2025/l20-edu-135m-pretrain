# RLVR on 135M Then 1B

## Research Question

Can RLVR with GSM8K verifiable rewards produce measurable reasoning gains at
135M parameters, and where does the effect become reliable as we scale toward
500M and 1B?

## Why This Route

RLVR is attractive here because the reward is programmatic and auditable: for
GSM8K, the final numeric answer can be extracted and compared with the gold
answer. GRPO is the practical default because it samples multiple completions
per prompt and normalizes rewards within the group, avoiding a separate learned
reward model.

The experiment should be framed as a scaling study, not just a single model
tuning run:

| Stage | Model | Goal | Stop/Go Criteria |
| --- | --- | --- | --- |
| C0 | 135M | Verify data, exact-answer eval, and reward signal | nonzero sampled pass rate and stable reward logs |
| C1 | 135M | Short GRPO run on GSM8K train | higher GSM8K exact accuracy without six-task mean collapse |
| C2 | 135M | Ablate reward shaping and generations per prompt | identify stable recipe and failure modes |
| C3 | 500M/1B | Repeat best recipe | show whether RLVR effect scales with parameter count |

## 135M Recipe

- Base: selected Stage 4 SFT/interpolated checkpoint.
- Dataset: GSM8K train for RLVR prompts; GSM8K test held out for exact-answer
  evaluation.
- Reward: `1.0` for exact final numeric answer plus a small `0.0-0.3` format
  shaping reward when an answer is extractable and the response is nontrivial.
- Algorithm: TRL `GRPOTrainer`.
- Initial conservative settings for one L20:
  - `num_generations=4`
  - `max_prompt_length=384`
  - `max_completion_length=320`
  - Correctness reward plus final-answer formatting reward, with repeated
    sentence/ngram penalties to avoid degenerate self-copying.
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=4`
  - `learning_rate=2e-6`
  - `beta=0.02`
  - `max_steps=250` for first real run

## Required Measurements

- GSM8K exact-answer accuracy before and after RLVR.
- Sampled pass@k on a fixed prompt subset before and after RLVR.
- Six-task regression eval after RLVR: ARC-Challenge, ARC-Easy, HellaSwag,
  LAMBADA OpenAI, PIQA, WinoGrande.
- Reward mean/std, completion length, clipped/truncated completion ratio.
- Qualitative failure buckets: arithmetic slip, wrong extraction, repeated
  reasoning, no final answer, memorized-format answer.

## Implementation Entrypoints

```bash
pip install -e '.[eval,rlvr]'
python scripts/prepare_gsm8k_rlvr_data.py
bash scripts/run_rlvr_gsm8k_135m.sh
```

The run script performs data preparation, pre-RLVR GSM8K eval, GRPO training,
and post-RLVR GSM8K eval. After the post-RLVR eval passes, run the existing
six-task benchmark script against the RLVR checkpoint.

## 1B Expansion

Only scale after C1 shows a real positive signal. The 1B run should reuse the
same prompt/reward/eval code and vary only model size plus memory parameters.
This keeps the paper question clean: whether the same verifiable reward recipe
starts working more reliably as parameter count increases.
