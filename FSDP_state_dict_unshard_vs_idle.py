"""
FSDP2 state_dict: idle/sharded vs during unshard window.

  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_state_dict_unshard_vs_idle.py
  # other terminal: node_rank=1, same port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def _describe_sd(rank: int, tag: str, sd: dict) -> None:
    for k, v in sd.items():
        if isinstance(v, DTensor):
            local = v.to_local()
            print(
                f"[rank {rank}] {tag} {k}: DTensor global={tuple(v.shape)} "
                f"local={tuple(local.shape)} placements={v.placements} "
                f"local_sum={local.float().sum().item():.4f}"
            )
        elif torch.is_tensor(v):
            print(
                f"[rank {rank}] {tag} {k}: Tensor shape={tuple(v.shape)} "
                f"sum={v.float().sum().item():.4f}"
            )
        else:
            print(f"[rank {rank}] {tag} {k}: type={type(v).__name__}")


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    with torch.no_grad():
        model.lin.weight.copy_(
            torch.arange(16 * 16, dtype=torch.float32).view(16, 16)
        )
    model = model.to(device)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)

    # --- idle / sharded ---
    print(f"[rank {rank}] === idle (sharded) ===")
    w = model.lin.weight
    if isinstance(w, DTensor):
        print(
            f"[rank {rank}] param local={tuple(w.to_local().shape)} "
            f"placements={w.placements}"
        )
    sd_idle = {k: v.detach().clone() if torch.is_tensor(v) else v
               for k, v in model.state_dict().items()}
    _describe_sd(rank, "idle_sd", sd_idle)

    # --- unshard window ---
    print(f"[rank {rank}] === unshard window ===")
    model.unshard()
    w = model.lin.weight
    if isinstance(w, DTensor):
        print(
            f"[rank {rank}] param local={tuple(w.to_local().shape)} "
            f"placements={w.placements}"
        )
    else:
        print(f"[rank {rank}] param Tensor shape={tuple(w.shape)}")

    sd_full = {k: v.detach().clone() if torch.is_tensor(v) else v
               for k, v in model.state_dict().items()}
    _describe_sd(rank, "unshard_sd", sd_full)

    model.reshard()
    print(f"[rank {rank}] === after reshard ===")
    w = model.lin.weight
    if isinstance(w, DTensor):
        print(
            f"[rank {rank}] param local={tuple(w.to_local().shape)} "
            f"placements={w.placements}"
        )

    dist.barrier()
    if rank == 0:
        print("OK: state_dict idle vs unshard demo finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
