"""
FSDP2 unshard-window overwrite probe. MODE=healthy|poison

  MODE=poison torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_unshard_window_overwrite.py
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


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "poison")
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

    # Registered AFTER fully_shard so this runs after FSDP's unshard pre-hook.
    def _after_unshard(mod: nn.Module, _inputs: tuple) -> None:
        w = mod.lin.weight
        if isinstance(w, DTensor):
            full = w.full_tensor()
            print(
                f"[rank {rank}] after unshard full_sum={full.sum().item():.4f} "
                f"local_sum={w.to_local().sum().item():.4f}"
            )
        if mode == "poison" and rank == 1:
            with torch.no_grad():
                # Overwrite the all-gathered full weight on this rank only.
                if isinstance(w, DTensor):
                    w.to_local().fill_(999.0)
                    # also poke through full view if still writable
                    try:
                        w.full_tensor().fill_(999.0)
                    except Exception as e:
                        print(f"[rank {rank}] full_tensor fill skipped: {type(e).__name__}")
                else:
                    w.fill_(999.0)
            print(f"[rank {rank}] poisoned weight during unshard window")

    model.register_forward_pre_hook(_after_unshard)

    x = torch.randn(4, 16, device=device)
    dist.broadcast(x, src=0)
    y = model(x)
    print(f"[rank {rank}] out.sum={y.sum().item():.4f}")

    w = model.lin.weight
    if isinstance(w, DTensor):
        print(
            f"[rank {rank}] after fwd local_sum={w.to_local().sum().item():.4f}"
        )

    dist.barrier()
    if rank == 0:
        print(f"OK: unshard-window demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
