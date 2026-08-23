#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TRANSLATION_SYSTEM_PROMPT = """Translate only the source request delimited by <request> and </request> into natural Simplified Chinese. Preserve code, numbers, explicit constraints, and any requested answer language or output format. Output only the translated request, with no explanation, labels, or quotation marks."""

ANSWER_SYSTEM_PROMPT = """Answer the user's request accurately and concisely. Follow every explicit language and formatting constraint. Keep the answer under 250 Chinese characters unless code or an explicit format requires more."""


def load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def shard_jobs_by_length(
    jobs: list[dict[str, Any]], *, rank: int, world_size: int
) -> list[dict[str, Any]]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    ordered = sorted(
        jobs,
        key=lambda job: (-len(str(job.get("prompt_en") or "")), str(job.get("id") or "")),
    )
    shards: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for job in ordered:
        target = min(range(world_size), key=lambda index: (loads[index], index))
        shards[target].append(job)
        loads[target] += len(str(job.get("prompt_en") or ""))
    return shards[rank]


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_teacher_output(text: str) -> tuple[dict[str, str] | None, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    tagged = re.fullmatch(
        r"\s*<user>(.*?)</user>\s*<assistant>(.*?)</assistant>\s*",
        text,
        flags=re.DOTALL,
    )
    if tagged is not None:
        user, assistant = tagged.groups()
    else:
        value = extract_json_object(text)
        if value is None:
            return None, "invalid_protocol"
        if set(value) != {"user", "assistant"}:
            return None, "invalid_keys"
        user = value["user"]
        assistant = value["assistant"]
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None, "non_string_value"
    user = user.strip()
    assistant = assistant.strip()
    if not user or not assistant or user == assistant:
        return None, "empty_or_identical"
    if len(user) > 6_000 or len(assistant) > 12_000:
        return None, "too_long"
    if len(re.findall(r"[\u4e00-\u9fff]", user)) < 4:
        return None, "user_not_chinese"
    if "<think>" in assistant or "</think>" in assistant:
        return None, "thinking_leak"
    return {"user": user, "assistant": assistant}, "ok"


def batches(values: list[Any], batch_size: int) -> list[list[Any]]:
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def read_completed_ids(*paths: Path) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    result.add(str(json.loads(line)["id"]))
    return result


def clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def clean_translation(text: str) -> str:
    text = clean_generated_text(text)
    wrapped = re.fullmatch(r"\s*<request>\s*(.*?)\s*</request>\s*", text, re.DOTALL)
    return wrapped.group(1).strip() if wrapped is not None else text


def generate_batch(
    model: Any,
    tokenizer: Any,
    chats: list[list[dict[str, str]]],
    *,
    device: torch.device,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for chat in chats
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    prefix_length = inputs["input_ids"].shape[1]
    return [
        clean_generated_text(text)
        for text in tokenizer.batch_decode(
            generated[:, prefix_length:], skip_special_tokens=True
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bilingual SFT data with Qwen3.")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("teacher generation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / f"teacher-shard-{rank:02d}-of-{world_size:02d}.jsonl"
    partial_path = final_path.with_suffix(".jsonl.partial")
    rejection_path = args.output_dir / f"rejected-shard-{rank:02d}-of-{world_size:02d}.jsonl"
    if final_path.exists():
        completed_rows = len(read_completed_ids(final_path))
        print(
            json.dumps(
                {"event": "already_complete", "rank": rank, "rows": completed_rows}
            )
        )
        return
    all_jobs = load_jobs(args.jobs)
    if args.max_jobs is not None:
        if args.max_jobs <= 0:
            raise ValueError("max jobs must be positive")
        all_jobs = all_jobs[: args.max_jobs]
    shard_jobs = shard_jobs_by_length(all_jobs, rank=rank, world_size=world_size)
    completed = read_completed_ids(partial_path)
    pending = [job for job in shard_jobs if job["id"] not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    started = time.monotonic()
    accepted = rejected = 0
    mode = "a" if partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8") as output, rejection_path.open(
        "a", encoding="utf-8"
    ) as rejection_output:
        for batch_index, job_batch in enumerate(batches(pending, args.batch_size), start=1):
            translation_chats = [
                [
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"<request>\n{job['prompt_en']}\n</request>",
                    },
                ]
                for job in job_batch
            ]
            translations = [
                clean_translation(text)
                for text in generate_batch(
                    model,
                    tokenizer,
                    translation_chats,
                    device=device,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                )
            ]
            answer_jobs: list[tuple[dict[str, Any], str]] = []
            for job, translation in zip(job_batch, translations, strict=True):
                if len(re.findall(r"[\u4e00-\u9fff]", translation)) < 4:
                    rejected += 1
                    rejection_output.write(
                        json.dumps(
                            {
                                "id": job["id"],
                                "stage": "translation",
                                "reason": "user_not_chinese",
                                "raw": translation,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue
                answer_jobs.append((job, translation))
            if not answer_jobs:
                continue
            answer_chats = [
                [
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": translation},
                ]
                for _, translation in answer_jobs
            ]
            answers = generate_batch(
                model,
                tokenizer,
                answer_chats,
                device=device,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            for (job, translation), answer in zip(answer_jobs, answers, strict=True):
                if not answer or len(answer) > 12_000:
                    rejected += 1
                    rejection_output.write(
                        json.dumps(
                            {
                                "id": job["id"],
                                "stage": "answer",
                                "reason": "empty_or_too_long",
                                "raw": answer,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue
                row = {
                    "id": job["id"],
                    "source": f"qwen3-8b-zh:{job['source']}",
                    "source_digest": job["source_digest"],
                    "teacher_model": "Qwen/Qwen3-8B",
                    "messages": [
                        {"role": "user", "content": translation},
                        {"role": "assistant", "content": answer},
                    ],
                }
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
            output.flush()
            rejection_output.flush()
            if batch_index == 1 or batch_index % 10 == 0:
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "rank": rank,
                            "accepted": accepted,
                            "rejected": rejected,
                            "pending_total": len(pending),
                            "examples_per_second": (accepted + rejected) / max(elapsed, 1e-9),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    partial_path.replace(final_path)
    print(
        json.dumps(
            {
                "event": "done",
                "rank": rank,
                "assigned": len(shard_jobs),
                "accepted_new": accepted,
                "rejected_new": rejected,
                "output": str(final_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
