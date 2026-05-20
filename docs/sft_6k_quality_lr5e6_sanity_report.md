# SFT Sanity Eval

Automatic checks passed: 3 / 5

## capital_china

**Instruction**

Answer with only the city name: What is the capital of China?

**Response**

```text
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
The capital of China is Beijing.
```

Score: `contains_expected` / `True`

## capital_uk

**Instruction**

Answer with only the city name: What is the capital of the United Kingdom?

**Response**

```text
The capital of the United Kingdom is London.
The capital of the United Kingdom is New York.
The capital of the United Kingdom is London.
The capital of the United Kingdom is New York.
The capital of the United Kingdom is London.
The capital of the United Kingdom is New York.
The capital of the United Kingdom is New York.
The capital of the United Kingdom is New York.
The capital of the United Kingdom is New York.
```

Score: `contains_expected` / `True`

## capital_uae

**Instruction**

Answer with only the city name: What is the capital of the United Arab Emirates?

**Response**

```text
The capital of the United Arab Emirates is Abu Dhabi.
The capital of the United Arab Emirates is Abu Dhabi.
The capital of the United Arab Emirates is Abu Dhabi.
The capital of the United Arab Emirates is Abu Dhabi.
The capital of the United Arab Emirates is Abu Dhabi.
The capital of the United Arab Emirates is Abu Dhabi.
The capital of the United Arab Emirates is Abu Dhabi
```

Score: `contains_expected` / `True`

## capital_new_zealand

**Instruction**

Answer with only the city name: What is the capital of New Zealand?

**Response**

```text
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is New Zealand.
The capital of New Zealand is
```

Score: `missing_expected` / `False`

## json_format

**Instruction**

Return valid JSON with keys "city" and "country" for Paris, France.

**Response**

```text
Return valid JSON with keys "city" and "country" for Paris, France.
### System:
Return valid JSON with keys "city" and "country" for Paris, France.
### Response:
Return valid JSON with keys "city" and "country" for Paris, France.
### System:
Return valid JSON with keys "city" and "country" for Paris, France.
### Response:
Return valid JSON with keys "city"
```

Score: `invalid_json` / `False`

## short_story

**Instruction**

Write a 5 sentence story about a student training a tiny language model.

**Response**

```text
The student was given a 5-minute presentation on the topic of the week.
### System:
The student was given a 5-minute presentation on the topic of the week.
### Response:
The student was given a 5-minute presentation on the topic of the week.
### System:
The student was given a 5-minute presentation on the topic of the week.
### Response:
The student was given a 5-
```

Score: `manual_review` / `None`

## concise_explanation

**Instruction**

Explain in two short bullet points why supervised fine-tuning masks prompt tokens.

**Response**

```text
Explain why the use of a mask is a good way to prevent the spread of COVID-19.
### System:
Explain why the use of masks is a good way to prevent the spread of COVID-19.
### System:
Explain why the use of masks is a good way to prevent the spread of COVID-19.
### Response:
Explain why the use of masks is a good way to prevent the spread of COVID-19.
```

Score: `manual_review` / `None`
