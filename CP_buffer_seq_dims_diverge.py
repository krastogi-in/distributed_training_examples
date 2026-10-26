"""
CP buffer_seq_dims diverge probe. MODE=healthy|dims

  # What: every rank must shard buffers on the SAME sequence axis.
  # Healthy: both use seq dim 1 for [B, S, H].
  # dims:    rank 0 uses dim 1, rank 1 uses dim 2 (wrong axis).

  MODE=dims torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_buffer_seq_dims_diverge.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "dims")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    mesh = init_device_mesh(device, (world_size,))
    torch.manual_seed(0)

    # [batch, seq, hidden] — real sequence axis is dim 1.
    B, S, H = 1, 8, 16
    assert S % world_size == 0
    assert H % world_size == 0  # so dim-2 "shard" can also run without remainder

    tokens = torch.randn(B, S, H, device=device)
    dist.broadcast(tokens, src=0)

    if mode == "healthy" or rank == 0:
        seq_dim = 1
    else:
        seq_dim = 2

    print(
        f"[rank {rank}] MODE={mode} buffer_seq_dims=[{seq_dim}] "
        f"before={tuple(tokens.shape)}"
    )

    with context_parallel(mesh, buffers=[tokens], buffer_seq_dims=[seq_dim]):
        print(f"[rank {rank}] inside CP tokens={tuple(tokens.shape)}")
        print(f"[rank {rank}] local sum={tokens.float().sum().item():.4f}")

    print(f"[rank {rank}] after CP tokens={tuple(tokens.shape)}")

    dist.barrier()
    if rank == 0:
        print(f"OK: CP seq-dims demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
