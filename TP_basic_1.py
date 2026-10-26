"""
Hands-on 02: Tensor Parallel (Colwise + Rowwise) on a tiny MLP.

  torchrun --standalone --nnodes=1 --nproc-per-node=2 TP_basic_1.py

Watch:
  - weight shard shapes after parallelize_module
  - forward output matches (replicated) across ranks
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    mesh = init_device_mesh("cpu", (world_size,))
    model = TinyMLP()
    # Same init on all ranks before sharding
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    plan = {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    model = parallelize_module(model, mesh, plan)

    w1 = model.fc1.weight
    w2 = model.fc2.weight
    print(
        f"[rank {rank}] fc1.weight local={tuple(w1.to_local().shape)} "
        f"placements={w1.placements}"
    )
    print(
        f"[rank {rank}] fc2.weight local={tuple(w2.to_local().shape)} "
        f"placements={w2.placements}"
    )

    x = torch.randn(2, 8)
    dist.broadcast(x, src=0)
    y = model(x)
    print(f"[rank {rank}] out shape={tuple(y.shape)} sum={y.sum().item():.4f}")

    dist.barrier()
    if rank == 0:
        print("OK: hands_on_02 TP finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
