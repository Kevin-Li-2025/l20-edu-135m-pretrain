from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_throughput_matrix import summarize_root


def write_log(path: Path, rates: list[float]) -> None:
    path.parent.mkdir(parents=True)
    events = [
        {
            "event": "start",
            "world_size": 5,
            "block_size": 2048,
            "tokens_per_step": 983040,
            "ddp_bucket_cap_mb": 25,
            "ddp_static_graph": True,
            "ddp_gradient_compression": "none",
        }
    ]
    events.extend(
        {
            "event": "train",
            "step": step,
            "tokens_per_sec_window": rate,
            "mfu_pct": 20 + step,
        }
        for step, rate in zip((5, 10, 15, 20), rates, strict=True)
    )
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")


def test_summarize_root_uses_steady_state_median(tmp_path: Path) -> None:
    write_log(tmp_path / "baseline-2k-5g" / "train.log", [1, 100, 110, 120])
    write_log(tmp_path / "candidate" / "train.log", [1, 120, 130, 140])
    (tmp_path / "candidate" / "telemetry.csv").write_text(
        "2026/08/22 00:00:00, 0, 100, 100, 44000, 220, 1305\n"
        "2026/08/22 00:00:01, 0, 80, 100, 44000, 100, 900\n",
        encoding="utf-8",
    )

    result = summarize_root(tmp_path, min_step=10)

    assert result["best_case"] == "candidate"
    assert result["cases"]["baseline-2k-5g"]["tokens_per_sec_median"] == 110
    assert result["cases"]["candidate"]["delta_vs_baseline_pct"] == pytest.approx(
        100 * (130 / 110 - 1)
    )
    telemetry = result["cases"]["candidate"]["gpu_telemetry_at_or_above_90pct_util"]
    assert telemetry["0"]["samples"] == 1
    assert telemetry["0"]["power_w_mean"] == 220
