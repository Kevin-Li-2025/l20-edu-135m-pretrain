from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audits_complete_partition(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    teacher_dir = tmp_path / "teacher"
    write_jsonl(jobs, [{"id": "a"}, {"id": "b"}])
    write_jsonl(
        teacher_dir / "teacher-shard-00-of-01.jsonl",
        [
            {
                "id": "a",
                "source": "qwen3-8b-zh:test",
                "messages": [
                    {"role": "user", "content": "请准确回答这个问题。"},
                    {"role": "assistant", "content": "答案。"},
                ],
            }
        ],
    )
    write_jsonl(
        teacher_dir / "rejected-shard-00-of-01.jsonl",
        [{"id": "b", "reason": "user_not_chinese"}],
    )
    out = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_bilingual_teacher_sft.py",
            "--jobs",
            str(jobs),
            "--teacher-dir",
            str(teacher_dir),
            "--world-size",
            "1",
            "--out",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["accepted"] == 1
    assert payload["rejected"] == 1
    assert payload["acceptance_rate"] == 0.5
