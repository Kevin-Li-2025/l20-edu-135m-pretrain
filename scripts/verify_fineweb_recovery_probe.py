"""Verify the archived engineering probes; never infer full-training quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_HASHES = {
    "probe-1560874_0.out": "7b0d2285f685bec54b5d965e0114b4fe4b92b38862422824c59da7aea897dd7b",
    "probe-1560874_1.out": "fa042198509dc592408c94946c3d284326a17cc0cd0e48c2b4d54d2c64d6821f",
    "probe-1560874_0.err": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "probe-1560874_1.err": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
EXPECTED = {
    "deep_cosine-1560876.json": (
        "c2de046ea1677b1b986afbb76d94d8b79da1eae38139b6d5b011f17cd8a7d6c3",
        "deep_cosine",
        134515008,
    ),
    "wide_cosine-1560874.json": (
        "bbf2671e104ff28f31b81caa09f3fa6e265265700c6de680234799f781256cc8",
        "wide_cosine",
        141576960,
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_receipt(receipt: dict, role: str, parameters: int, root: Path) -> None:
    require(receipt["status"] == "PASS", "probe did not pass")
    require(
        receipt["scope"] == "memory_and_optimizer_smoke_not_quality_or_speed",
        "scope mismatch",
    )
    config_path = f"configs/fineweb_recovery/{role}_s20260906.yaml"
    require(receipt["config"] == config_path, "configuration role mismatch")
    require(
        hashlib.sha256((root / config_path).read_bytes()).hexdigest()
        == receipt["config_sha256"],
        "configuration hash mismatch",
    )
    require(receipt["gpu"] == "NVIDIA GeForce RTX 4090", "GPU mismatch")
    require(receipt["parameters"] == parameters, "parameter count mismatch")
    require(receipt["seed"] == 20260906, "seed mismatch")
    require(receipt["tokens_per_step"] == 159744, "token budget mismatch")
    require(
        receipt["sampled_parameters_changed"] is True, "no verified optimizer update"
    )
    require(receipt["validation_smoke_batches"] == 1, "unexpected validation coverage")
    require(math.isfinite(receipt["validation_smoke_loss"]), "nonfinite validation")
    steps = receipt["steps"]
    require([row["step"] for row in steps] == [1, 2, 3], "incomplete optimizer steps")
    for row in steps:
        require(math.isfinite(row["loss"]), "nonfinite training loss")
        require(
            math.isfinite(row["grad_norm"]) and row["grad_norm"] > 0, "invalid gradient"
        )
    require(
        0
        < receipt["peak_allocated_bytes"]
        <= receipt["peak_reserved_bytes"]
        < receipt["total_device_bytes"],
        "invalid memory accounting",
    )


def verify(directory: Path, root: Path = ROOT) -> None:
    for name, digest in LOG_HASHES.items():
        require(
            hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest,
            f"archived log changed: {name}",
        )
    manifest = directory / "EXECUTION_SHA256SUMS"
    require(
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        == "e0e1205238c8f081e99f43b47066e2eae82750cfa8f55c2bbe0b897e1a6fef63",
        "execution manifest changed",
    )
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(
            hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected,
            f"frozen execution file changed: {relative}",
        )
    for name, (digest, role, parameters) in EXPECTED.items():
        payload = (directory / name).read_bytes()
        require(
            hashlib.sha256(payload).hexdigest() == digest,
            f"archived receipt changed: {name}",
        )
        validate_receipt(json.loads(payload), role, parameters, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.directory)
    print("verified: two GPU engineering probes; no full-training or quality claim")
