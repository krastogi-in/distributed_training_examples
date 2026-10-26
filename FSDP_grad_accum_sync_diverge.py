"""
FSDP2 grad-accum sync / reshard_after_backward probe.
MODE=healthy|sync|reshard_bwd

  MODE=sync torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_grad_accum_sync_diverge.py
  # other terminal: node_rank=1, same MODE/port
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
    mode = os.environ.get("MODE", "sync")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_micro = 2

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
    for micro in range(num_micro):
        is_last = micro == num_micro - 1

        if mode == "healthy":
            # Sync grads only on last microbatch; same on every rank.
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(True)
        elif mode == "sync":
            # Rank 0 matches healthy; rank 1 always syncs every microbatch.
            if rank == 0:
                model.set_requires_gradient_sync(is_last)
            else:
                model.set_requires_gradient_sync(True)
            model.set_reshard_after_backward(True)
        else:
            # reshard_bwd: agree on sync schedule; diverge reshard_after_backward.
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(rank == 0)

        print(
            f"[rank {rank}] micro={micro} is_last={is_last} "
            f"grad_sync={is_last if mode != 'sync' or rank == 0 else True} "
            f"reshard_after_bwd={True if mode != 'reshard_bwd' else (rank == 0)}"
        )

        x = torch.randn(4, 16, device=device)
        dist.broadcast(x, src=0)
        # Distinct microbatch contents after broadcast base.
        x = x + float(micro)
        y = model(x)
        loss = y.float().sum() / num_micro
        loss.backward()
        print(f"[rank {rank}] micro={micro} loss={loss.item():.4f}")

    g = model.lin.weight.grad
    if g is None:
        print(f"[rank {rank}] grad is None")
    elif isinstance(g, DTensor):
        print(
            f"[rank {rank}] grad local_sum={g.to_local().float().sum().item():.4f}"
        )
    else:
        print(f"[rank {rank}] grad sum={g.float().sum().item():.4f}")

    opt.step()
    w1 = model.lin.weight.to_local().detach().sum().item()
    print(f"[rank {rank}] local_sum after={w1:.4f} delta={w1 - w0:.4f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: grad-accum demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
