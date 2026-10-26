"""
FSDP2 same-shape swap probe. MODE=healthy|order|submods

  MODE=order torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_same_shape_swap.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class TwinLinears(nn.Module):
    """Two linears, identical shapes — easy to mis-pair under FSDP."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16, bias=False)
        self.b = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def _tag(rank: int, name: str, p: nn.Parameter) -> None:
    if isinstance(p, DTensor):
        local = p.to_local()
        print(
            f"[rank {rank}] {name}: DTensor local_sum={local.sum().item():.4f} "
            f"local={tuple(local.shape)} placements={p.placements}"
        )
    else:
        print(
            f"[rank {rank}] {name}: Tensor sum={p.detach().sum().item():.4f} "
            f"shape={tuple(p.shape)}"
        )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device, (world_size,))

    model = TwinLinears()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    
    # Distinguishable fills so a mis-pair shows up in local_sum.
    with torch.no_grad():
        model.a.weight.fill_(1.0)
        model.b.weight.fill_(2.0)
    model = model.to(device)
    if rank == 0:
        fully_shard(model.a, mesh=mesh,reshard_after_forward=True)
        fully_shard(model.b, mesh=mesh,reshard_after_forward=True)
        fully_shard(model, mesh=mesh,reshard_after_forward=True)
    else:
        fully_shard(model.a, mesh=mesh,reshard_after_forward=False)
        fully_shard(model.b, mesh=mesh,reshard_after_forward=False)
        fully_shard(model, mesh=mesh,reshard_after_forward=False)


    _tag(rank, "a.weight", model.a.weight)
    _tag(rank, "b.weight", model.b.weight)
    for i in range(10):    
        x = torch.randn(4, 16)
        dist.broadcast(x, src=0)
        x = x.to(device)
        y = model(x)
        print(f"[rank {rank}] out.sum={y.sum().item():.4f}")


if __name__ == "__main__":
    main()
