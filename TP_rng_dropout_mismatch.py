"""
TP RNG probe (CPU + gloo). MODE=healthy|diverge

  MODE=diverge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 TP_rng_dropout_mismatch.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
    mode = os.environ.get("MODE", "diverge")
    dist.init_process_group(backend="gloo")

    # Agreed param init — isolate forward RNG, not weight broadcast.
    torch.manual_seed(0)
    mesh = init_device_mesh("cpu", (world_size,))
    model = TinyMLP()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model = parallelize_module(
        model,
        mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )

    x = torch.randn(4, 8)
    dist.broadcast(x, src=0)

    if mode == "healthy":
        torch.manual_seed(123)
    else:
        torch.manual_seed(123 if rank == 0 else 999)

    model.train()
    y = model(x)
    # Dropout on replicated TP output.
    y = F.dropout(y, p=0.5, training=True)

    print(
        f"[rank {rank}] MODE={mode} out.sum={y.sum().item():.4f} "
        f"out.mean={y.mean().item():.4f} nonzero={(y != 0).sum().item()}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: TP rng demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
