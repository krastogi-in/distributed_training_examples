"""
FSDP2 data-triggered desync probe. MODE=healthy|uneven|early_exit

  MODE=uneven torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_data_triggered_desync.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "uneven")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model = model.to(device)
    fully_shard(model, mesh=mesh)

    # Simulate per-rank dataset length / microbatch count (no Join).
    if mode == "healthy":
        num_steps = 3
    elif mode == "uneven":
        num_steps = 3 if rank == 0 else 2
    else:
        # early_exit: both start with 3, rank 1 breaks after first step
        num_steps = 3

    print(f"[rank {rank}] MODE={mode} planned_steps={num_steps}")

    for step in range(num_steps):
        if mode == "early_exit" and rank == 1 and step >= 1:
            print(f"[rank {rank}] early exit at step={step}")
            break

        x = torch.randn(4, 16, device=device)
        # Do NOT broadcast x — data-dependent path; only step count matters here.
        y = model(x)
        loss = y.float().sum()
        loss.backward()
        print(f"[rank {rank}] step={step} loss={loss.item():.4f}")

    print(f"[rank {rank}] loop done")
    dist.barrier()
    if rank == 0:
        print(f"OK: data-desync demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
