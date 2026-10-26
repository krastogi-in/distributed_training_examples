"""
TP: SequenceParallel on vs off for the same LayerNorm (CPU + gloo).

Isolates SP — no Colwise/Rowwise (those shard *features*, which breaks LN(16)).

Rank 0: SequenceParallel() on norm  → treats input as sequence-sharded
Rank 1: plain LayerNorm             → expects full sequence

  # terminal A
  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 \
      TP_sequence_parallel_mismatch.py

  # terminal B
  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 \
      TP_sequence_parallel_mismatch.py

What to record yourself:
  - norm.weight DTensor vs plain
  - x_in shape each rank passes
  - hang / error / wrong out at forward
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import SequenceParallel, parallelize_module


class NormOnly(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        return self.norm(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    mesh = init_device_mesh("cpu", (world_size,))
    model = NormOnly(hidden=8)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    if rank == 0:
        plan = {"norm": SequenceParallel(sequence_dim=1)}
    else:
        plan = {}  # SequenceParallel off

    print(f"[rank {rank}] plan={plan}")
    model = parallelize_module(model, mesh, plan)

    w = model.norm.weight
    if hasattr(w, "placements"):
        print(
            f"[rank {rank}] norm.weight local={tuple(w.to_local().shape)} "
            f"placements={w.placements}"
        )
    else:
        print(f"[rank {rank}] norm.weight plain shape={tuple(w.shape)}")

    # Full activation both ranks start from (same broadcast).
    x_full = torch.randn(2, 4, 8)  # [batch, seq, hidden]
    dist.broadcast(x_full, src=0)

    # Intentional contract mismatch:
    # - SP (rank 0) assumes local input is already sharded on seq dim
    # - plain LN (rank 1) uses the full sequence
    if rank == 0:
        x_in = x_full.chunk(world_size, dim=1)[rank].contiguous()
    else:
        x_in = x_full

    print(
        f"[rank {rank}] x_full={tuple(x_full.shape)} x_in={tuple(x_in.shape)} "
        f"calling forward"
    )
    y = model(x_in)
    print(
        f"[rank {rank}] out shape={tuple(y.shape)} "
        f"sum={y.detach().float().sum().item():.4f}"
    )

    dist.barrier()
    if rank == 0:
        print("OK: SequenceParallel mismatch demo finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
