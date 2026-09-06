"""Three real optimizer steps and one validation batch; engineering smoke only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from l20_pretrain.config import load_config
from l20_pretrain.modeling import count_parameters
from l20_pretrain.train import (
    autocast_context,
    build_loader,
    evaluate,
    get_dtype,
    load_or_create_model,
    load_tokenizer,
    make_optimizer,
    make_scheduler,
    move_batch,
    preflight_training_data,
    preflight_validation_data,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one allocated CUDA device")
    name = torch.cuda.get_device_name(0)
    if "4090" not in name:
        raise RuntimeError(f"expected RTX 4090, got {name}")
    config = load_config(args.config)
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    device, dtype = torch.device("cuda"), get_dtype(config.trainer.dtype)
    tokenizer = load_tokenizer(config)
    preflight_training_data(config, tokenizer)
    preflight_validation_data(config, tokenizer)
    model = load_or_create_model(config, tokenizer, None, dtype).to(device)
    model.train()
    optimizer = make_optimizer(model, config, device)
    scheduler = make_scheduler(optimizer, config)
    iterator = iter(build_loader(config, tokenizer))
    first_parameter = next(model.parameters())
    initial = first_parameter.detach().flatten()[:1024].cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(config.trainer.gradient_accumulation_steps):
            batch = move_batch(next(iterator), device)
            with autocast_context(device, dtype):
                loss = model(**batch).loss / config.trainer.gradient_accumulation_steps
            losses.append(float(loss.detach().cpu()))
            loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_clip)
        )
        if (
            not math.isfinite(grad_norm)
            or grad_norm <= 0
            or not all(map(math.isfinite, losses))
        ):
            raise RuntimeError(
                "nonfinite loss/gradient or zero gradient in training probe"
            )
        optimizer.step()
        scheduler.step()
        row = {"step": step, "loss": math.fsum(losses), "grad_norm": grad_norm}
        steps.append(row)
        print(json.dumps(row), flush=True)
    changed = not torch.equal(initial, first_parameter.detach().flatten()[:1024].cpu())
    if not changed:
        raise RuntimeError("sampled parameters did not change after optimizer updates")
    eval_config = replace(config, trainer=replace(config.trainer, eval_batches=1))
    eval_loss = evaluate(model, eval_config, tokenizer, device, dtype)
    if not math.isfinite(eval_loss):
        raise RuntimeError("nonfinite validation smoke loss")
    torch.cuda.synchronize()
    payload = {
        "status": "PASS",
        "scope": "memory_and_optimizer_smoke_not_quality_or_speed",
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "gpu": name,
        "parameters": count_parameters(model),
        "seed": config.seed,
        "tokens_per_step": config.tokens_per_step,
        "steps": steps,
        "sampled_parameters_changed": changed,
        "validation_smoke_loss": eval_loss,
        "validation_smoke_batches": 1,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "total_device_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
