"""
TP plan-keys probe (CPU + gloo). MODE=healthy|keys

  MODE=keys torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 TP_plan_keys_diverge.py
  # other terminal: node_rank=1, same MODE/port
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
    mode = os.environ.get("MODE", "keys")
    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    mesh = init_device_mesh("cpu", (world_size,))
    model = TinyMLP()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    if mode == "healthy":
        plan = {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    elif rank == 0:
        plan = {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    else:
        plan = {"fc1": ColwiseParallel()}

    print(f"[rank {rank}] MODE={mode} plan_keys={sorted(plan.keys())}")
    model = parallelize_module(model, mesh, plan)

    for name, mod in [("fc1", model.fc1), ("fc2", model.fc2)]:
        w = mod.weight
        kind = type(w).__name__
        if hasattr(w, "to_local"):
            print(
                f"[rank {rank}] {name} {kind} local={tuple(w.to_local().shape)} "
                f"placements={w.placements}"
            )
        else:
            print(f"[rank {rank}] {name} {kind} shape={tuple(w.shape)}")

    x = torch.randn(2, 8)
    dist.broadcast(x, src=0)
    y = model(x)
    print(f"[rank {rank}] out sum={y.sum().item():.4f} shape={tuple(y.shape)}")

    dist.barrier()
    if rank == 0:
        print(f"OK: TP plan-keys demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
