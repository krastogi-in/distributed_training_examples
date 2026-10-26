"""
FSDP2 explicit unshard() order probe. MODE=healthy|order

  MODE=order torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_unshard_order_diverge.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16, bias=False)
        self.b = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "order")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    model = Twin()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    with torch.no_grad():
        model.a.weight.fill_(1.0)
        model.b.weight.fill_(2.0)
    model = model.to(device)

    fully_shard(model.a, mesh=mesh)
    fully_shard(model.b, mesh=mesh)
    fully_shard(model, mesh=mesh)

    print(f"[rank {rank}] MODE={mode}")

    # Explicit unshard order — same-sized modules, so order must match.
    if mode == "healthy" or rank == 0:
        print(f"[rank {rank}] unshard a then b")
        model.a.unshard()
        model.b.unshard()
    else:
        print(f"[rank {rank}] unshard b then a")
        model.b.unshard()
        model.a.unshard()

    x = torch.randn(4, 16, device=device)
    dist.broadcast(x, src=0)
    y = model(x)
    print(f"[rank {rank}] out.sum={y.sum().item():.4f}")

    model.a.reshard()
    model.b.reshard()

    dist.barrier()
    if rank == 0:
        print(f"OK: unshard-order demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
