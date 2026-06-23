from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

from .data import tokenize_without_specials

IGNORE_INDEX = -100
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful, accurate, concise AI assistant."
    "<|im_end|>\\n' }}"
    "{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] "
    "+ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


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
    if not isinstance(example, dict):
        return None

    messages = _parse_messages(example.get(messages_column))
    if not messages:
        instruction = _text(example.get(instruction_column)) or _text(example.get(prompt_column))
        extra_input = _text(example.get(input_column))
        response = _text(example.get(output_column)) or _text(example.get(response_column))
        if not instruction or not response:
            return None
        user_content = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]

    if system_prompt and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt.strip()}, *messages]

    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            continue
        segment = (
            f"<|im_start|>{role}\n"
            f"{message['content'].strip()}<|im_end|>\n"
        )
        segment_ids = tokenize_without_specials(tokenizer, segment)
        if not segment_ids:
            continue
        input_ids.extend(segment_ids)
        if train_on_prompt or role == "assistant":
            labels.extend(segment_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(segment_ids))

    if not input_ids:
        return None

    input_ids = input_ids[:block_size]
    labels = labels[:block_size]
    if not any(label != IGNORE_INDEX for label in labels):
        return None

    attention_mask = [1] * len(input_ids)

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


class PackedSFTTokenDataset(IterableDataset):
    def __init__(self, dataset: SFTTokenDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.block_size = dataset.block_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        input_buffer: list[int] = []
        label_buffer: list[int] = []
        emitted_supervised_tokens = 0

        for encoded in self.dataset:
            input_ids = encoded["input_ids"].tolist()
            labels = encoded["labels"].tolist()
            if not input_ids or not any(label != IGNORE_INDEX for label in labels):
                continue

            cursor = 0
            while cursor < len(input_ids):
                remaining = self.block_size - len(input_buffer)
                if remaining <= 0:
                    yield self._pack(input_buffer, label_buffer)
                    input_buffer = []
                    label_buffer = []
                    emitted_supervised_tokens = 0
                    remaining = self.block_size

                take = min(remaining, len(input_ids) - cursor)
                input_buffer.extend(input_ids[cursor : cursor + take])
                label_buffer.extend(labels[cursor : cursor + take])
                emitted_supervised_tokens += sum(
                    1 for label in labels[cursor : cursor + take] if label != IGNORE_INDEX
                )
                cursor += take

                if len(input_buffer) >= self.block_size:
                    if emitted_supervised_tokens > 0:
                        yield self._pack(input_buffer, label_buffer)
                    input_buffer = []
                    label_buffer = []
                    emitted_supervised_tokens = 0

        if input_buffer and emitted_supervised_tokens > 0:
            yield self._pack(input_buffer, label_buffer)

    @staticmethod
    def _pack(input_ids: list[int], labels: list[int]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_sft_batch(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(int(row["input_ids"].shape[0]) for row in rows)
    padded_length = ((max_length + 7) // 8) * 8

    def pad(tensor: torch.Tensor, value: int) -> torch.Tensor:
        padding = padded_length - int(tensor.shape[0])
        if padding <= 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, padding), value=value)

    return {
        "input_ids": torch.stack([pad(row["input_ids"], 0) for row in rows], dim=0),
        "attention_mask": torch.stack(
            [pad(row["attention_mask"], 0) for row in rows], dim=0
        ),
        "labels": torch.stack(
            [pad(row["labels"], IGNORE_INDEX) for row in rows], dim=0
        ),
    }
