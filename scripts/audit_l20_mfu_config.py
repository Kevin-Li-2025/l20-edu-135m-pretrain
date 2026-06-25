#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model") or {}
    trainer = config.get("trainer") or {}
    block_size = int(model.get("block_size") or 0)
    micro_batch = int(trainer.get("micro_batch_size") or 0)
    grad_accum = int(trainer.get("gradient_accumulation_steps") or 1)
    tokens_per_step = block_size * micro_batch * grad_accum
    findings: list[dict[str, str]] = []
    recommendations: list[str] = []

    if block_size > 4096:
        findings.append(
            {
                "severity": "high",
                "issue": "block_size_above_4k",
                "detail": "8K main training spends too much L20 compute on quadratic attention for this 135M target.",
            }
        )
        recommendations.append("Use 2048 as the main throughput phase and reserve 4096 for late curriculum.")
    elif block_size == 4096:
        findings.append(
            {
                "severity": "medium",
                "issue": "4k_context_tradeoff",
                "detail": "4K is useful for curriculum, but measured 2K throughput was materially higher.",
            }
        )
        recommendations.append("Benchmark a 2048 context phase with micro_batch_size 16 before long 4K runs.")

    if not bool(trainer.get("liger_kernel")):
        findings.append(
            {
                "severity": "high",
                "issue": "liger_disabled",
                "detail": "Prior L20 measurements selected SDPA + Liger + compile as the fastest short-context setup.",
            }
        )
        recommendations.append("Enable trainer.liger_kernel for pretraining and SFT benchmarks.")

    if not bool(trainer.get("compile")):
        findings.append(
            {
                "severity": "medium",
                "issue": "compile_disabled",
                "detail": "Compile improved the best recorded 2K benchmark, though it should be validated per host.",
            }
        )
        recommendations.append("Benchmark compile=true for the selected micro-batch before long runs.")

    if bool(trainer.get("gradient_checkpointing")) and block_size <= 4096:
        findings.append(
            {
                "severity": "medium",
                "issue": "checkpointing_cost",
                "detail": "Gradient checkpointing saves memory but usually lowers tokens/sec when the run fits without it.",
            }
        )
        recommendations.append("Disable gradient checkpointing unless the selected micro-batch OOMs.")

    attn = str(model.get("attn_implementation") or "")
    if attn not in {"sdpa", "flash_attention_2"}:
        findings.append(
            {
                "severity": "medium",
                "issue": "attention_backend_unspecified",
                "detail": "Explicit SDPA or flash_attention_2 makes benchmarks comparable.",
            }
        )
        recommendations.append("Set model.attn_implementation to sdpa or flash_attention_2.")

    if micro_batch < 8 and block_size <= 4096:
        findings.append(
            {
                "severity": "medium",
                "issue": "small_micro_batch",
                "detail": "Low micro-batch may underfill the L20 for a 135M model.",
            }
        )
        recommendations.append("Benchmark micro_batch_size 12 and 16 at 2K; use the largest stable value.")

    if tokens_per_step < 32768:
        findings.append(
            {
                "severity": "low",
                "issue": "low_tokens_per_step",
                "detail": "Very small token batches can reduce hardware utilization and make logging noisy.",
            }
        )
        recommendations.append("Increase micro-batch or gradient accumulation after validating loss stability.")

    return {
        "status": "pass" if not any(item["severity"] == "high" for item in findings) else "review",
        "block_size": block_size,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "tokens_per_step": tokens_per_step,
        "attn_implementation": attn or None,
        "compile": bool(trainer.get("compile")),
        "liger_kernel": bool(trainer.get("liger_kernel")),
        "gradient_checkpointing": bool(trainer.get("gradient_checkpointing")),
        "findings": findings,
        "recommendations": sorted(set(recommendations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit L20 135M training config for MFU/tokens/sec risks.")
    parser.add_argument("config")
    parser.add_argument("--out")
    args = parser.parse_args()
    payload = audit_config(load_yaml(Path(args.config)))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
