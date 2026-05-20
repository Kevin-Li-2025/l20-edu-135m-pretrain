# SFT Sanity Eval

Automatic checks passed: 3 / 5

## capital_china

**Instruction**

Answer with only the city name: What is the capital of China?

**Response**

```text
Sure, here are some answers to the question: "What is the capital of China?" The capital of China is Beijing. It is located in the country of China. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing. The capital of China is Beijing
```

Score: `contains_expected` / `True`

## capital_uk

**Instruction**

Answer with only the city name: What is the capital of the United Kingdom?

**Response**

```text
Sure, here are some answers to your question:
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is London.
- The capital of the United Kingdom is
```

Score: `contains_expected` / `True`

## capital_uae

**Instruction**

Answer with only the city name: What is the capital of the United Arab Emirates?

**Response**

```text
Sure, here are some answers to your question:
- The capital of the United Arab Emirates is Abu Dhabi.
- The capital of the United Arab Emirates is Abu Dhabi.
- The capital of the United Arab Emirates is Abu Dhabi.
- The capital of the United Arab Emirates is Abu Dhabi.
- The capital of the United Arab Emirates is Abu Dhabi.
- The capital of the United Arab Emirates is Abu
```

Score: `contains_expected` / `True`

## capital_new_zealand

**Instruction**

Answer with only the city name: What is the capital of New Zealand?

**Response**

```text
The capital of New Zealand is New Zealand. It is located in the South Island of New Zealand. It is the 10th largest country in the world. It is also the 10th largest country in the world by area. The capital city is New Zealand.
The capital city is New Zealand. It is located in the South Island of New Zealand. It is the 10th largest country in the world by area. The capital city is New Zealand
```

Score: `missing_expected` / `False`

## json_format

**Instruction**

Return valid JSON with keys "city" and "country" for Paris, France.

**Response**

```text
Of course, you can use the following commands to create a new JSON object:
>>> import JSON >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "Paris" >>> city = "
```

Score: `invalid_json` / `False`

## short_story

**Instruction**

Write a 5 sentence story about a student training a tiny language model.

**Response**

```text
Of course! Here are some examples of how to write a 5 sentence story about a student training a tiny language model:
- "I was a little bit nervous about the training, but I was able to learn a lot of new things." - "I was able to learn a lot of new things from the training, but I was able to learn a lot of new things from the training." - "I was able to learn a lot of new things from the training
```

Score: `manual_review` / `None`

## concise_explanation

**Instruction**

Explain in two short bullet points why supervised fine-tuning masks prompt tokens.

**Response**

```text
Sure, here are some examples of supervised fine-tuning masks:
- The first example is a simple one: "The first time I saw the first token, I was shocked to see the first token. I was shocked to see the first token, but I was surprised to see the first token. I was shocked to see the first token, but I was surprised to see the first token. I was shocked to see the first token, but I was surprised to see the
```

Score: `manual_review` / `None`
