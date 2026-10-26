"""
FSDP2 grad reduce-scatter probe. MODE=healthy|zero_grad

  MODE=zero_grad torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_skip_reduce_scatter.py
  # other terminal: node_rank=1, same MODE/port

Note: Parameter.register_hook does NOT fire on FSDP2 DTensor weights (verified).
We zero this rank's contribution by scaling the loss so local grads are 0;
FSDP still runs reduce-scatter and both ranks continue.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "zero_grad")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    with torch.no_grad():
        model.lin.weight.fill_(1.0)
    model = model.to(device)
    fully_shard(model, mesh=mesh)
    opt = optim.SGD(model.parameters(), lr=0.1)

    w0 = model.lin.weight.to_local().detach().sum().item()
    print(f"[rank {rank}] MODE={mode} local_sum before={w0:.4f}")

    opt.zero_grad()
    x = torch.randn(4, 16, device=device)
    dist.broadcast(x, src=0)
    y = model(x)
    loss = y.float().sum()
    print(f"[rank {rank}] out.sum={loss.item():.4f}")

    if mode == "zero_grad" and rank == 1:
        # Local grads become 0; reduce-scatter still runs (peer contributes).
        loss = loss * 0.0
        print(f"[rank {rank}] scaled loss -> 0 (zero local grad contribution)")

    loss.backward()

    g = model.lin.weight.grad
    if g is None:
        print(f"[rank {rank}] grad is None after backward")
    elif isinstance(g, DTensor):
        print(
            f"[rank {rank}] grad after bwd local_sum={g.to_local().float().sum().item():.4f} "
            f"placements={g.placements}"
        )
    else:
        print(f"[rank {rank}] grad after bwd sum={g.float().sum().item():.4f}")

    opt.step()
    w1 = model.lin.weight.to_local().detach().sum().item()
    print(f"[rank {rank}] local_sum after={w1:.4f} delta={w1 - w0:.4f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: reduce-scatter probe finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
