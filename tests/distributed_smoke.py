from __future__ import annotations

from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from l20_pretrain.config import PretrainConfig
from l20_pretrain.train import (
    cleanup_distributed,
    initialize_distributed,
    wrap_distributed_model,
)


def main() -> None:
    context = initialize_distributed(os.environ.get("SMOKE_DEVICE", "cpu"))
    try:
        torch.manual_seed(1234)
        model = torch.nn.Linear(8, 4).to(context.device)
        model = wrap_distributed_model(model, context, PretrainConfig())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(2):
            sync = model.no_sync() if micro_step == 0 else nullcontext()
            with sync:
                inputs = torch.full(
                    (3, 8),
                    float(context.rank + micro_step + 1),
                    device=context.device,
                )
                model(inputs).square().mean().div(2).backward()
        optimizer.step()

        parameter = next(model.parameters()).detach().flatten()
        gathered = [torch.empty_like(parameter) for _ in range(context.world_size)]
        dist.all_gather(gathered, parameter)
        assert all(torch.equal(gathered[0], candidate) for candidate in gathered[1:])
        if context.is_main:
            print("distributed smoke passed", flush=True)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
