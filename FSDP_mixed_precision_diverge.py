"""
FSDP2 MixedPrecisionPolicy probe. MODE=healthy|param|reduce  BACKEND=gloo|nccl

  BACKEND=gloo MODE=param torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_mixed_precision_diverge.py
  # other terminal: node_rank=1, same MODE/BACKEND/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "param")
    backend = os.environ.get("BACKEND", "gloo")

    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("BACKEND=nccl needs CUDA")
    if backend == "gloo":
        device = "cpu"
    else:
        device = "cuda"

    dist.init_process_group(backend=backend)
    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    # Baseline policy both ranks use in MODE=healthy.
    param_dtype = torch.float16
    reduce_dtype = torch.float32
    if mode == "param" and rank == 1:
        param_dtype = torch.float32
    if mode == "reduce" and rank == 1:
        reduce_dtype = torch.float16

    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    print(
        f"[rank {rank}] MODE={mode} BACKEND={backend} "
        f"param_dtype={param_dtype} reduce_dtype={reduce_dtype}"
    )

    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model = model.to(device)
    fully_shard(model, mesh=mesh, mp_policy=mp)

    x = torch.randn(4, 16, device=device)
    dist.broadcast(x, src=0)
    y = model(x)
    loss = y.float().sum()
    loss.backward()

    w = model.lin.weight
    g = w.grad
    print(
        f"[rank {rank}] out.sum={y.float().sum().item():.4f} "
        f"w.dtype={w.dtype} grad_is_none={g is None}"
    )
    if g is not None:
        local = g.to_local() if hasattr(g, "to_local") else g
        print(
            f"[rank {rank}] grad.dtype={g.dtype} "
            f"grad.local_sum={local.float().sum().item():.4f}"
        )

    dist.barrier()
    if rank == 0:
        print(f"OK: mp-policy demo finished (MODE={mode} BACKEND={backend})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
