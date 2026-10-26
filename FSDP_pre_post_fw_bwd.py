"""
FSDP2 same-shape swap probe. MODE=healthy|fwd_order|submods

  MODE=fwd_order torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
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
            f"[rank {rank}] {name}: local_sum={local.sum().item():.4f} "
            f"local={tuple(local.shape)}"
        )
    else:
        print(
            f"[rank {rank}] {name}: sum={p.detach().sum().item():.4f} "
            f"shape={tuple(p.shape)}"
        )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "fwd_order")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device, (world_size,))

    model = TwinLinears()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    with torch.no_grad():
        model.a.weight.fill_(1.0)
        model.b.weight.fill_(2.0)
    model = model.to(device)
    
    fully_shard(model.a, mesh=mesh)
    fully_shard(model.b, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # def _pre_hook(mod: nn.Module, _inputs: tuple) -> None:
    #     name = "a" if mod is model.a else "b"
    #     mod.weight.fill_(0.0)
    #     w = mod.weight
    #     if isinstance(w, DTensor):
    #         # full_tensor() forces the all-gather result into view
    #         print(
    #             f"[rank {rank}] pre {name}: full_sum={w.full_tensor().sum().item():.4f} "
    #             f"local_sum={w.to_local().sum().item():.4f}"
    #         )
    
    # if rank == 0:
    #     model.b.register_forward_pre_hook(_pre_hook)
    #     model.a.register_forward_pre_hook(_pre_hook)

    if rank == 0:
        model._forward_pre_hooks.clear()
        model._forward_hooks.clear()
        model.a._forward_pre_hooks.clear()
        model.b._forward_pre_hooks.clear()
        model.a._forward_hooks.clear()
        model.b._forward_hooks.clear()

    _tag(rank, "a.weight idle", model.a.weight)
    _tag(rank, "b.weight idle", model.b.weight)

    x = torch.randn(4, 16)
    dist.broadcast(x, src=0)
    x = x.to(device)
    y = model(x)
    print(f"[rank {rank}] out.sum={y.sum().item():.4f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: FSDP same-shape demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
