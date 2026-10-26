"""
TP loss all-reduce probe (CPU + gloo). MODE=healthy|skip|local|poison

  MODE=healthy torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 TP_loss_allreduce_mismatch.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._functional_collectives import all_reduce as functional_all_reduce
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
    mode = os.environ.get("MODE", "healthy")
    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    mesh = init_device_mesh("cpu", (world_size,))
    model = TinyMLP()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # Agreed TP plan — isolate the loss contract, not plan diverge.
    model = parallelize_module(
        model,
        mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )
    opt = optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(4, 8)
    target = torch.randn(4, 8)
    dist.broadcast(x, src=0)
    dist.broadcast(target, src=0)

    w0 = model.fc1.weight.to_local().detach().sum().item()
    print(f"[rank {rank}] MODE={mode} fc1.local_sum before={w0:.4f}")

    opt.zero_grad()
    y = model(x)
    loss = nn.functional.mse_loss(y, target)
    print(f"[rank {rank}] local loss (pre-reduce)={loss.item():.6f}")

    if mode == "local":
        if rank == 1:
            loss = loss * 10.0
            print(f"[rank {rank}] LOCAL-ONLY 10x loss={loss.item():.6f} (no all_reduce)")
        else:
            print(f"[rank {rank}] LOCAL-ONLY loss={loss.item():.6f} (no all_reduce)")
    elif mode == "skip" and rank == 1:
        print(f"[rank {rank}] SKIPPING loss all_reduce")
    else:
        to_reduce = loss
        if mode == "poison" and rank == 1:
            to_reduce = loss + 999.0
            print(f"[rank {rank}] poisoned local loss={to_reduce.item():.6f}")
        shared = functional_all_reduce(to_reduce, "sum", dist.group.WORLD) / world_size
        print(f"[rank {rank}] loss after all_reduce/avg={shared.item():.6f}")
        if mode == "poison":
            loss = loss * (shared.detach() / (loss.detach() + 1e-8))
            print(f"[rank {rank}] train loss (shared scale)={loss.item():.6f}")
        else:
            loss = shared

    loss.backward()
    opt.step()

    w1 = model.fc1.weight.to_local().detach().sum().item()
    print(f"[rank {rank}] fc1.local_sum after={w1:.4f} delta={w1 - w0:.4f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: loss-allreduce demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
