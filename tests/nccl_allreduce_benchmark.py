from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure NCCL all-reduce bandwidth.")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[4, 64, 256, 512])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        for size_mb in args.sizes_mb:
            element_count = size_mb * 1024 * 1024 // 2
            tensor = torch.ones(element_count, dtype=torch.bfloat16, device=device)
            for _ in range(args.warmup):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            dist.barrier()

            started = time.perf_counter()
            for _ in range(args.iterations):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

            if rank == 0:
                seconds_per_collective = elapsed_tensor.item() / args.iterations
                algorithm_gbps = size_mb / 1024 / seconds_per_collective
                bus_gbps = algorithm_gbps * 2 * (world_size - 1) / world_size
                print(
                    f"size_mb={size_mb} time_ms={seconds_per_collective * 1000:.3f} "
                    f"algbw_GBps={algorithm_gbps:.3f} busbw_GBps={bus_gbps:.3f}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
