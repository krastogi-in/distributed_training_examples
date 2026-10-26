"""
FSDP2 CPUOffloadPolicy / pin_memory probe. MODE=healthy|pin|offload

  MODE=pin torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_cpu_offload_pin_diverge.py
  # other terminal: node_rank=1, same MODE/port
  # needs CUDA
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "pin")

    if not torch.cuda.is_available():
        raise RuntimeError("CPU offload probe needs CUDA")

    dist.init_process_group(backend="nccl")
    torch.manual_seed(0)
    device = "cuda"
    mesh = init_device_mesh(device, (world_size,))

    if mode == "healthy":
        offload = CPUOffloadPolicy(pin_memory=True)
    elif mode == "pin":
        offload = CPUOffloadPolicy(pin_memory=(rank == 0))
    else:
        # offload: rank 0 offloads, rank 1 does not
        offload = (
            CPUOffloadPolicy(pin_memory=True) if rank == 0 else OffloadPolicy()
        )

    print(f"[rank {rank}] MODE={mode} offload_policy={offload!r}")

    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model = model.to(device)
    fully_shard(model, mesh=mesh, offload_policy=offload)

    x = torch.randn(4, 16, device=device)
    dist.broadcast(x, src=0)
    y = model(x)
    loss = y.sum()
    loss.backward()

    w = model.lin.weight
    print(
        f"[rank {rank}] out.sum={y.sum().item():.4f} "
        f"w.device={w.device} w.dtype={w.dtype}"
    )
    if w.grad is not None:
        g = w.grad
        print(f"[rank {rank}] grad.device={g.device}")

    dist.barrier()
    if rank == 0:
        print(f"OK: cpu-offload demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
