"""
TP init probe (CPU + gloo). MODE=healthy|skip

  MODE=skip torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 TP_init_divergent.py
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
    mode = os.environ.get("MODE", "skip")
    dist.init_process_group(backend="gloo")

    if rank == 0:
        torch.manual_seed(0)

    mesh = init_device_mesh("cpu", (world_size,))
    model = TinyMLP()

    pre = {n: p.detach().sum().item() for n, p in model.named_parameters()}
    print(
        f"[rank {rank}] MODE={mode} pre-bcast "
        f"fc1.sum={pre['fc1.weight']:.4f} fc2.sum={pre['fc2.weight']:.4f}"
    )

    if mode == "healthy" or rank == 0:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        post = {n: p.detach().sum().item() for n, p in model.named_parameters()}
        print(
            f"[rank {rank}] after broadcast "
            f"fc1.sum={post['fc1.weight']:.4f} fc2.sum={post['fc2.weight']:.4f}"
        )
    else:
        print(f"[rank {rank}] skip broadcast")

    model = parallelize_module(
        model,
        mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )

    w1 = model.fc1.weight.to_local().detach()
    w2 = model.fc2.weight.to_local().detach()
    print(
        f"[rank {rank}] after TP fc1.local_sum={w1.sum().item():.4f} "
        f"fc2.local_sum={w2.sum().item():.4f}"
    )

    x = torch.randn(2, 8)
    dist.broadcast(x, src=0)
    y = model(x)
    print(f"[rank {rank}] out sum={y.sum().item():.4f} shape={tuple(y.shape)}")

    dist.barrier()
    if rank == 0:
        print(f"OK: TP init demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
