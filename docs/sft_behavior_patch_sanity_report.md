# SFT Sanity Eval

Automatic checks passed: 3 / 5

## capital_china

**Instruction**

Answer with only the city name: What is the capital of China?

**Response**

```text
The capital of China is Beijing. It is located in the north of the country. The capital city is Beijing.
The capital of China is Beijing. It is located in the south of the country. The capital city is Beijing.
The capital of China is Beijing. It is located in the north of the country. The capital city is Beijing.
The capital of China is Beijing. It is located in the south of the country. The capital city is Beijing.
```

Score: `contains_expected` / `True`

## capital_uk

**Instruction**

Answer with only the city name: What is the capital of the United Kingdom?

**Response**

```text
The capital of the United Kingdom is London. It is located in the north of England. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London. The capital city is London.
```

Score: `contains_expected` / `True`

## capital_uae

**Instruction**

Answer with only the city name: What is the capital of the United Arab Emirates?

**Response**

```text
The capital of the United Arab Emirates is Abu Dhabi. It is located in the country's southern region, and is the second-largest city in the country. The city is located in the Arabian Peninsula, and is bordered by the Arabian Sea to the north, the Arabian Gulf to the east, and the Arabian Sea to the south. The city is surrounded by the Arabian Sea to the west, the Arabian Sea to the east, and the Arabian Sea to the north.
```

Score: `contains_expected` / `True`

## capital_new_zealand

**Instruction**

Answer with only the city name: What is the capital of New Zealand?

**Response**

```text
The capital of New Zealand is New Zealand. It is located in the South Island of New Zealand. It is the 10th largest country in the world. It is also the 10th largest country in the world by area.
The capital of New Zealand is New Zealand. It is located in the South Island of New Zealand. It is the 10th largest country in the world by area.
The capital of New Zealand is New Zealand. It
```

Score: `missing_expected` / `False`

## json_format

**Instruction**

Return valid JSON with keys "city" and "country" for Paris, France.

**Response**

```text
Paris, France, 2012-01-01 10:40:00 GMT
Paris, France, 2012-01-01 10:40:00 GMT
Paris, France, 2012-01-01 10:40:00 GMT
Paris, France, 2012-01-01 10
```

Score: `invalid_json` / `False`

## short_story

**Instruction**

Write a 5 sentence story about a student training a tiny language model.

**Response**

```text
The student training a tiny language model is a great way to help your students learn a new language. Here are some tips for using this model:
- Use a story to help your students learn a new language.
- Use a story to help your students learn a new language.
- Use a story to help your students learn a new language.
- Use a story to help your students learn a new language.
- Use a story to help your students learn a
```

Score: `manual_review` / `None`

## concise_explanation

**Instruction**

Explain in two short bullet points why supervised fine-tuning masks prompt tokens.

**Response**

```text
The first point is that supervised fine-tuning masks are used to prevent the use of tokens. They are used to prevent the use of tokens in the production of tokens. The second point is that supervised fine-tuning masks are used to prevent the use of tokens in the production of tokens. They are used to prevent the use of tokens in the production of tokens. The third point is that supervised fine-tuning masks are used to prevent the use of tokens in the production of tokens
```

Score: `manual_review` / `None`
