"""
Hands-on 03: FSDP2 fully_shard lifecycle (unshard around forward).

  torchrun --standalone --nnodes=1 --nproc-per-node=2 hands_on_03_fsdp2.py

Watch:
  - before forward: param is DTensor sharded
  - during/after hooks: unshard for compute, reshard after
  - each rank's local shard numel ~ 1/world
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def _describe(rank: int, tag: str, p: nn.Parameter) -> None:
    if isinstance(p, DTensor):
        local = p.to_local()
        print(
            f"[rank {rank}] {tag}: DTensor placements={p.placements} "
            f"global={tuple(p.shape)} local={tuple(local.shape)} "
            f"local_numel={local.numel()}"
        )
    else:
        print(
            f"[rank {rank}] {tag}: Tensor shape={tuple(p.shape)} numel={p.numel()}"
        )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # FSDP2 prefers NCCL+CUDA; on CPU-only boxes use gloo (limited).
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.manual_seed(0)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device_type, (world_size,))

    model = TinyBlock()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model = model.to(device_type)

    # Bottom-up style: shard the block (one group).
    # reshard_after_forward=True so post-forward frees full weights again.
    fully_shard(model, mesh=mesh, reshard_after_forward=True)

    _describe(rank, "after fully_shard (idle/sharded)", model.lin.weight)

    x = torch.randn(4, 16)
    dist.broadcast(x, src=0)
    x = x.to(device_type)

    # Calling the module triggers pre_forward unshard
    y = model(x)
    _describe(rank, "after forward (often reshared)", model.lin.weight)
    print(f"[rank {rank}] out sum={y.sum().item():.4f}")

    # Explicit unshard/reshard APIs if available
    if hasattr(model, "unshard"):
        model.unshard()
        _describe(rank, "after model.unshard()", model.lin.weight)
        model.reshard()
        _describe(rank, "after model.reshard()", model.lin.weight)

    dist.barrier()
    if rank == 0:
        print("OK: hands_on_03 FSDP2 finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
