#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file


COPY_PATTERNS = (
    "*.json",
    "*.model",
    "*.txt",
    "*.yaml",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
)


def model_file(directory: Path) -> Path:
    files = sorted(directory.glob("*.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one safetensors file in {directory}, found {len(files)}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate a base and SFT checkpoint to reduce alignment forgetting."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    base_dir = Path(args.base).resolve()
    sft_dir = Path(args.sft).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_file(model_file(base_dir), device="cpu")
    sft = load_file(model_file(sft_dir), device="cpu")
    if base.keys() != sft.keys():
        missing_base = sorted(sft.keys() - base.keys())
        missing_sft = sorted(base.keys() - sft.keys())
        raise RuntimeError(
            f"Checkpoint keys differ: missing_base={missing_base[:5]}, "
            f"missing_sft={missing_sft[:5]}"
        )

    merged: dict[str, torch.Tensor] = {}
    for name, base_tensor in base.items():
        sft_tensor = sft[name]
        if base_tensor.shape != sft_tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: {base_tensor.shape} != {sft_tensor.shape}"
            )
        if torch.is_floating_point(base_tensor):
            tensor = torch.lerp(
                base_tensor.float(),
                sft_tensor.float(),
                args.alpha,
            ).to(sft_tensor.dtype)
        else:
            tensor = sft_tensor.clone()
        merged[name] = tensor.contiguous()

    save_file(
        merged,
        output_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    for pattern in COPY_PATTERNS:
        for source in sft_dir.glob(pattern):
            if source.is_file() and source.name != "model.safetensors":
                shutil.copy2(source, output_dir / source.name)
    (output_dir / "interpolation.json").write_text(
        json.dumps(
            {
                "base": str(base_dir),
                "sft": str(sft_dir),
                "alpha": args.alpha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
