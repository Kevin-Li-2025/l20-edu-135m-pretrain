from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_bilingual_teacher_sft.py"
SPEC = importlib.util.spec_from_file_location("generate_bilingual_teacher_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATE
SPEC.loader.exec_module(GENERATE)


def test_parse_teacher_output_accepts_strict_json() -> None:
    parsed, reason = GENERATE.parse_teacher_output(
        '{"user":"请计算二加二。","assistant":"答案是四。"}'
    )

    assert reason == "ok"
    assert parsed == {"user": "请计算二加二。", "assistant": "答案是四。"}


def test_parse_teacher_output_accepts_tagged_protocol() -> None:
    parsed, reason = GENERATE.parse_teacher_output(
        "<user>请计算二加二。</user>\n<assistant>答案是四。</assistant>"
    )

    assert reason == "ok"
    assert parsed == {"user": "请计算二加二。", "assistant": "答案是四。"}


def test_parse_teacher_output_strips_thinking_and_fences() -> None:
    parsed, reason = GENERATE.parse_teacher_output(
        '<think>hidden</think>```json\n{"user":"请简要解释光合作用。","assistant":"植物利用光能合成有机物。"}\n```'
    )

    assert reason == "ok"
    assert parsed is not None
    assert "hidden" not in parsed["assistant"]


def test_parse_teacher_output_rejects_non_chinese_user() -> None:
    parsed, reason = GENERATE.parse_teacher_output(
        '{"user":"add two numbers","assistant":"答案是四。"}'
    )

    assert parsed is None
    assert reason == "user_not_chinese"


def test_clean_generated_text_removes_reasoning_and_outer_fence() -> None:
    actual = GENERATE.clean_generated_text(
        "<think>hidden</think>\n```text\n请只输出答案。\n```"
    )

    assert actual == "请只输出答案。"


def test_clean_translation_removes_request_delimiters() -> None:
    actual = GENERATE.clean_translation("<request>\n请只输出答案。\n</request>")

    assert actual == "请只输出答案。"


def test_length_aware_sharding_is_disjoint_and_balanced() -> None:
    jobs = [
        {"id": str(index), "prompt_en": "x" * length}
        for index, length in enumerate(range(1, 101))
    ]
    shards = [
        GENERATE.shard_jobs_by_length(jobs, rank=rank, world_size=5)
        for rank in range(5)
    ]

    assert sorted(job["id"] for shard in shards for job in shard) == sorted(
        job["id"] for job in jobs
    )
    loads = [sum(len(job["prompt_en"]) for job in shard) for shard in shards]
    assert max(loads) - min(loads) <= 20
