#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} is not hexadecimal") from exc


def verify_result(payload: dict[str, Any], *, repo_root: Path = ROOT) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported FineWeb result schema")

    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("the factorial receipt must contain exactly four cells")
    by_role = {cell.get("role"): cell for cell in cells}
    expected_roles = {"deep_thin_cosine", "deep_thin_wsd", "wide_cosine", "wide_wsd"}
    if set(by_role) != expected_roles:
        raise ValueError("factorial roles are missing or duplicated")

    completed = [cell for cell in cells if cell.get("state") == "COMPLETED"]
    failed = [cell for cell in cells if cell.get("state") == "FAILED"]
    if len(completed) != 2 or len(failed) != 2:
        raise ValueError("expected two completed and two failed cells")
    if any(cell.get("exit_code") != "0:0" for cell in completed):
        raise ValueError("completed cells must have a zero exit code")
    if any((cell.get("failure") or {}).get("type") != "CUDA_OUT_OF_MEMORY" for cell in failed):
        raise ValueError("both failed cells must preserve the CUDA OOM classification")

    for cell in cells:
        config_path = repo_root / str(cell["config"])
        expected = cell["config_sha256"]
        _require_sha256(expected, f"{cell['role']}.config_sha256")
        if sha256_file(config_path) != expected:
            raise ValueError(f"config digest mismatch: {config_path}")
        _require_sha256(cell["log_sha256"], f"{cell['role']}.log_sha256")
        _require_sha256(cell["stderr_sha256"], f"{cell['role']}.stderr_sha256")
        if cell["state"] == "COMPLETED":
            if cell.get("final_step") != payload["matched_controls"]["max_steps"]:
                raise ValueError(f"incomplete final step for {cell['role']}")
            for field in ("final_eval_loss", "final_eval_perplexity", "median_mfu_pct"):
                if not math.isfinite(float(cell[field])):
                    raise ValueError(f"non-finite {field} for {cell['role']}")
            for field in ("model_sha256", "trainer_state_sha256"):
                _require_sha256(cell["checkpoint"][field], f"{cell['role']}.{field}")
            _require_sha256(cell["telemetry"]["sha256"], f"{cell['role']}.telemetry")

    for relative_path, expected in payload["execution_source"]["files"].items():
        _require_sha256(expected, f"execution_source.{relative_path}")
        path = repo_root / relative_path
        if sha256_file(path) != expected:
            raise ValueError(f"execution source digest mismatch: {path}")

    cosine = by_role["wide_cosine"]
    wsd = by_role["wide_wsd"]
    comparison = payload["wide_schedule_comparison"]
    expected_loss_delta = wsd["final_eval_loss"] - cosine["final_eval_loss"]
    expected_ppl_delta = wsd["final_eval_perplexity"] - cosine["final_eval_perplexity"]
    expected_relative_ppl = 100.0 * (
        wsd["final_eval_perplexity"] / cosine["final_eval_perplexity"] - 1.0
    )
    checks = {
        "wsd_minus_cosine_eval_loss": expected_loss_delta,
        "wsd_minus_cosine_perplexity": expected_ppl_delta,
        "wsd_relative_perplexity_change_pct": expected_relative_ppl,
    }
    for field, expected in checks.items():
        if not math.isclose(float(comparison[field]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"comparison field is inconsistent: {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a committed FineWeb 1B receipt.")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    verify_result(payload, repo_root=args.repo_root.resolve())
    print(f"{args.receipt}: verified")


if __name__ == "__main__":
    main()
