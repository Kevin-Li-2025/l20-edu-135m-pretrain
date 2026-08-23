from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


NAMES = ("lr5e5", "lr1e4", "lr2e4", "lr3e4", "lr6e4")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_summarizes_all_five_candidates(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    eval_root = tmp_path / "eval"
    log_root.mkdir()
    for index, name in enumerate(NAMES):
        (log_root / f"{name}.log").write_text(
            "not-json\n"
            + json.dumps(
                {"event": "eval", "loss": 2.0 - index / 10, "perplexity": 7.0}
            )
            + "\n"
            + json.dumps({"event": "done", "checkpoint": f"/{name}"})
            + "\n",
            encoding="utf-8",
        )
        quality = {
            "metrics": {
                "prompt_pass_rate": index / 10,
                "check_pass_rate": 0.5,
                "stop_rate": 0.5,
                "degenerate_repetition_rate": 0.0,
            }
        }
        write_json(eval_root / name / "chat_quality.json", quality)
        write_json(eval_root / name / "zh_skill_quality.json", quality)
        write_json(
            eval_root / name / "ifeval" / "results_1.json",
            {
                "results": {
                    "ifeval": {
                        "prompt_level_strict_acc,none": 0.1,
                        "inst_level_strict_acc,none": 0.2,
                        "prompt_level_loose_acc,none": 0.3,
                        "inst_level_loose_acc,none": 0.4,
                        "sample_len": 541,
                    }
                }
            },
        )
        ci = [-0.01, 0.01] if name != "lr6e4" else [-0.03, -0.001]
        paired = {
            "aggregate": {
                "baseline_equal_task_mean": 0.42,
                "candidate_equal_task_mean": 0.421,
                "delta": 0.001,
                "paired_bootstrap_ci": ci,
            },
            "tasks": {
                "task": {
                    "paired_bootstrap_ci": ci,
                }
            },
        }
        write_json(eval_root / name / "paired-vs-base-primary.json", paired)
        write_json(eval_root / name / "paired-vs-base-development.json", paired)

    out = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_skill_sft_lr_pilot.py",
            "--log-root",
            str(log_root),
            "--eval-root",
            str(eval_root),
            "--out",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["candidates"]) == set(NAMES)
    assert payload["ranking"][0] == "lr3e4"
    assert "lr6e4" not in payload["retention_eligible"]
    assert payload["candidates"]["lr6e4"]["significant_task_regressions"] == [
        "task",
    ]
    assert payload["candidates"]["lr5e5"]["ifeval_samples"] == 541
