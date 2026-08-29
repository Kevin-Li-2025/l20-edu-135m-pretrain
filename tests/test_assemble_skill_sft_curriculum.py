from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "assemble_skill_sft_curriculum.py"
SPEC = importlib.util.spec_from_file_location("assemble_skill_sft_curriculum", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ASSEMBLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASSEMBLE
SPEC.loader.exec_module(ASSEMBLE)


def row(source: str, index: int) -> dict:
    messages = [
        {"role": "user", "content": f"prompt {index}"},
        {"role": "assistant", "content": "answer"},
    ]
    return {
        "source": source,
        "messages": messages,
        "digest": ASSEMBLE.canonical_digest(messages),
    }


def test_smoltalk_selection_is_order_independent() -> None:
    rows = [row("a", index) for index in range(10)] + [
        row("b", index + 10) for index in range(10)
    ]
    quotas = {"a": 3, "b": 2}

    first = ASSEMBLE.select_smoltalk_rows(rows, quotas, seed=9)
    second = ASSEMBLE.select_smoltalk_rows(reversed(rows), quotas, seed=9)

    assert sorted(item["digest"] for item in first) == sorted(
        item["digest"] for item in second
    )
    assert sum(item["source"] == "a" for item in first) == 3
    assert sum(item["source"] == "b" for item in first) == 2


def test_arithmetic_rows_are_deterministic_unique_and_bilingual() -> None:
    first = list(ASSEMBLE.arithmetic_rows(100, seed=12))
    second = list(ASSEMBLE.arithmetic_rows(100, seed=12))

    assert first == second
    assert len({row["digest"] for row in first}) == 100
    assert {row["source"] for row in first} == {
        "synthetic-arithmetic-en",
        "synthetic-arithmetic-zh",
    }


def test_gsm8k_reasoning_markup_is_removed() -> None:
    rows = list(
        ASSEMBLE.gsm8k_rows(
            [{"question": "What is 2+2?", "answer": "We add. <<2+2=4>> #### 4"}]
        )
    )

    assert len(rows) == 1
    assert "<<" not in rows[0]["messages"][1]["content"]
    assert rows[0]["messages"][1]["content"].endswith("Final answer: 4")


def test_short_exact_prompt_match_does_not_require_thirteen_tokens() -> None:
    candidate = {
        "messages": [
            {"role": "user", "content": "只回复绿色"},
            {"role": "assistant", "content": "绿色"},
        ]
    }
    index = {tuple(ASSEMBLE.normalize_tokens("只回复绿色")): "chat:zh"}

    assert ASSEMBLE.exact_prompt_match(candidate, index) == "chat:zh"


def test_teacher_rows_clean_only_boundary_protocol_tags(tmp_path: Path) -> None:
    path = tmp_path / "teacher-shard-00-of-01.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"<request>\\n请回答问题。\\n</request>"},'
        '{"role":"assistant","content":"答案。"}],"source":"teacher"}\n',
        encoding="utf-8",
    )
    stats = ASSEMBLE.Counter()

    rows = list(ASSEMBLE.teacher_rows([path], stats=stats))

    assert rows[0]["messages"][0]["content"] == "请回答问题。"
    assert stats["boundary_protocol_tags_cleaned"] == 2
