# Stage6 Post-Train Summary

Generated: `2026-06-27T11:51:54.023036+00:00`

## Training

- Latest step: `3050`
- Latest tokens: `299827200`
- Latest train loss: `2.9271`
- Latest tokens/s: `57693.9301`
- Latest MFU: `79.9718%`
- Latest eval loss: `2.7932`
- Latest eval perplexity: `16.3329`
- Stable mean MFU: `79.5166%`

## Six-Task Eval

- Status: `complete`
- Mean: `0.4150`

| Task | Metric | Score |
| --- | --- | --- |
| ARC-Challenge | acc_norm,none | 0.2867 |
| ARC-Easy | acc_norm,none | 0.4954 |
| HellaSwag | acc_norm,none | 0.3243 |
| LAMBADA OpenAI | acc,none | 0.2573 |
| PIQA | acc_norm,none | 0.6219 |
| WinoGrande | acc,none | 0.5043 |

## Nsight Compute Profile

- Status: `failed`
- Run dir: `logs/stage6-edu-reasoning/profile/tensor_profile_20260627_194843`
- Report: `logs/stage6-edu-reasoning/profile/tensor_profile_20260627_194843/stage6_tensor_profile.ncu-rep`

### Tensor-Core / Roofline Evidence Snippets

- `==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.`
