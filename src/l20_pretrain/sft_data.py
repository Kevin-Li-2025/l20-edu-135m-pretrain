from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

from .data import tokenize_without_specials

IGNORE_INDEX = -100


def iter_local_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("SFT jsonl rows must be JSON objects")
                yield value


class LocalJsonlExamples:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter_local_jsonl(self.path)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_messages(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None

    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = _text(item.get("role")).lower()
        content = _text(item.get("content"))
        if role and content:
            messages.append({"role": role, "content": content})
    return messages or None


def render_instruction_example(
    example: dict[str, Any],
    *,
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
    prompt_column: str = "prompt",
    response_column: str = "response",
    system_prompt: str | None = None,
) -> tuple[str, str] | None:
    instruction = _text(example.get(instruction_column)) or _text(example.get(prompt_column))
    extra_input = _text(example.get(input_column))
    response = _text(example.get(output_column)) or _text(example.get(response_column))
    if not instruction or not response:
        return None

    parts: list[str] = []
    if system_prompt:
        parts.append(f"### System:\n{system_prompt.strip()}\n\n")
    parts.append(f"### Instruction:\n{instruction}\n\n")
    if extra_input:
        parts.append(f"### Input:\n{extra_input}\n\n")
    parts.append("### Response:\n")
    return "".join(parts), response


def render_messages_example(
    example: dict[str, Any],
    *,
    messages_column: str = "messages",
    system_prompt: str | None = None,
) -> tuple[str, str] | None:
    messages = _parse_messages(example.get(messages_column))
    if not messages:
        return None

    assistant_indices = [idx for idx, msg in enumerate(messages) if msg["role"] == "assistant"]
    if not assistant_indices:
        return None
    target_index = assistant_indices[-1]
    response = messages[target_index]["content"]

    parts: list[str] = []
    if system_prompt:
        parts.append(f"### System:\n{system_prompt.strip()}\n\n")

    for msg in messages[:target_index]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"### System:\n{content}\n\n")
        elif role == "user":
            parts.append(f"### Instruction:\n{content}\n\n")
        elif role == "assistant":
            parts.append(f"### Response:\n{content}\n\n")

    parts.append("### Response:\n")
    return "".join(parts), response


def render_sft_example(
    example: Any,
    *,
    messages_column: str = "messages",
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
    prompt_column: str = "prompt",
    response_column: str = "response",
    system_prompt: str | None = None,
) -> tuple[str, str] | None:
    if not isinstance(example, dict):
        return None
    return render_messages_example(
        example,
        messages_column=messages_column,
        system_prompt=system_prompt,
    ) or render_instruction_example(
        example,
        instruction_column=instruction_column,
        input_column=input_column,
        output_column=output_column,
        prompt_column=prompt_column,
        response_column=response_column,
        system_prompt=system_prompt,
    )


def encode_sft_example(
    example: Any,
    tokenizer: Any,
    *,
    block_size: int,
    train_on_prompt: bool = False,
    messages_column: str = "messages",
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
    prompt_column: str = "prompt",
    response_column: str = "response",
    system_prompt: str | None = None,
) -> dict[str, torch.Tensor] | None:
    rendered = render_sft_example(
        example,
        messages_column=messages_column,
        instruction_column=instruction_column,
        input_column=input_column,
        output_column=output_column,
        prompt_column=prompt_column,
        response_column=response_column,
        system_prompt=system_prompt,
    )
    if rendered is None:
        return None

    prompt, response = rendered
    prompt_ids = tokenize_without_specials(tokenizer, prompt)
    response_ids = tokenize_without_specials(tokenizer, response)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        response_ids = [*response_ids, int(eos_token_id)]
    if not response_ids:
        return None

    input_ids = [*prompt_ids, *response_ids]
    if train_on_prompt:
        labels = input_ids.copy()
    else:
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids.copy()

    input_ids = input_ids[:block_size]
    labels = labels[:block_size]
    if not any(label != IGNORE_INDEX for label in labels):
        return None

    attention_mask = [1] * len(input_ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    padding = block_size - len(input_ids)
    if padding > 0:
        input_ids.extend([int(pad_token_id)] * padding)
        labels.extend([IGNORE_INDEX] * padding)
        attention_mask.extend([0] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTTokenDataset(IterableDataset):
    def __init__(
        self,
        examples: Iterable[Any],
        tokenizer: Any,
        *,
        block_size: int,
        max_examples: int | None = None,
        max_chars: int | None = None,
        train_on_prompt: bool = False,
        messages_column: str = "messages",
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
        prompt_column: str = "prompt",
        response_column: str = "response",
        system_prompt: str | None = None,
    ) -> None:
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_examples = max_examples
        self.max_chars = max_chars
        self.train_on_prompt = train_on_prompt
        self.messages_column = messages_column
        self.instruction_column = instruction_column
        self.input_column = input_column
        self.output_column = output_column
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.system_prompt = system_prompt

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        emitted = 0
        for example in self.examples:
            if self.max_chars is not None and len(json.dumps(example, ensure_ascii=False, default=str)) > self.max_chars:
                continue
            encoded = encode_sft_example(
                example,
                self.tokenizer,
                block_size=self.block_size,
                train_on_prompt=self.train_on_prompt,
                messages_column=self.messages_column,
                instruction_column=self.instruction_column,
                input_column=self.input_column,
                output_column=self.output_column,
                prompt_column=self.prompt_column,
                response_column=self.response_column,
                system_prompt=self.system_prompt,
            )
            if encoded is None:
                continue
            yield encoded
            emitted += 1
            if self.max_examples is not None and emitted >= self.max_examples:
                return


def collate_sft_batch(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows], dim=0),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows], dim=0),
        "labels": torch.stack([row["labels"] for row in rows], dim=0),
    }
