# Stage7 Ablation Plan

Stage6 was stable but flat on the six-task benchmark. Stage7 starts with a
single 50M-token trial before scaling any recipe.

## Trial A: SmolLM2-Style Rewarm

- Base: `runs/l20-stage6-edu-reasoning-300m/final`
- Tokens: 50M
- Context: 4096
- Training shape: micro batch 6, grad accumulation 4
- Schedule: re-warm to `5e-6`, cosine decay to 20%
- Replay: 20% Stage6 tokenized replay
- New data: DCLM-Edu, SmolLM fineweb-edu-dedup, StackExchange, PES2O,
  Wikipedia, FineMath, a small Cosmopedia tail, and a 2% Stack-Edu Python tail
- Guarding: cross-source guard plus 13-gram and LCS contamination filtering

Success criterion: the six-task mean must beat Stage6's `0.4150` without
obvious regression on PIQA or WinoGrande. If it fails, do not scale this recipe;
switch to a benchmark-aligned commonsense/distillation mixture.
