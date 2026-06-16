# Stage 4 Model Comparison

## Token And Benchmark Summary

| Model | Params | Reported tokens | Reported hardware | 6-task Mean | Budget context |
| --- | ---: | ---: | --- | ---: | --- |
| L20 Edu 135M Stage 4 | 134.5M | ~13B | 1x NVIDIA L20 | 0.4150 | 1.00x |
| SmolLM-135M | 135M | 600B | 64x H100 | 0.4767 | ~46.2x tokens |
| SmolLM2-135M | 135M | 2T | 64x H100 | 0.4917 | ~153.8x tokens |
| Qwen2.5-0.5B | 0.49B | not reported in HF card | not reported in HF card | 0.5363 | larger reference |
| OLMo-1B | 1B | 3T | not listed in HF card | 0.5681 | ~230.8x tokens; 1B upper bound |

## Six-Task Detail

| Task | Metric | L20 Edu 135M Stage 4 | SmolLM-135M | SmolLM2-135M | Qwen2.5-0.5B | OLMo-1B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | acc_norm,none | 0.2867 | 0.2875 | 0.2969 | 0.3225 | 0.3089 |
| ARC-Easy | acc_norm,none | 0.4958 | 0.5610 | 0.5854 | 0.5850 | 0.5711 |
| HellaSwag | acc_norm,none | 0.3240 | 0.4265 | 0.4301 | 0.5215 | 0.6294 |
| LAMBADA OpenAI | acc,none | 0.2602 | 0.3757 | 0.4289 | 0.5201 | 0.5539 |
| PIQA | acc_norm,none | 0.6148 | 0.6823 | 0.6839 | 0.6997 | 0.7481 |
| WinoGrande | acc,none | 0.5083 | 0.5272 | 0.5249 | 0.5691 | 0.5975 |

Notes:

- All benchmark numbers in this table are self-run with the same six-task lm-eval target suite and selected metrics.
- SmolLM-135M and SmolLM2-135M are the direct 135M public references; Qwen2.5-0.5B and OLMo-1B are larger reference/upper-bound checkpoints.
- Reported tokens and hardware are taken from public model cards where listed; Qwen2.5-0.5B does not list pretraining tokens or hardware in its Hugging Face card.
