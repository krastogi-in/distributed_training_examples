"""
CP seq-len / packing diverge probe. MODE=healthy|nondiv|diff_len

  What:
    CP shards sequence evenly: need S % world_size == 0 (unless padding/load
    balancer handles remainders). Also every rank must see the SAME seq_len
    from the data loader / packing. If rank0 has S=8 and rank1 has S=7,
    shard sizes and collectives disagree.

    healthy  — S=8, divisible, same on all ranks
    nondiv   — S=9, not divisible by world_size (same on all ranks)
    diff_len — rank0 S=8, rank1 S=6 (dataloader packing diverge)

  MODE=nondiv torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_seq_not_divisible_world_size.py
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
    mode = os.environ.get("MODE", "nondiv")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    mesh = init_device_mesh(device, (world_size,))
    torch.manual_seed(0)
    H = 16

    if mode == "healthy":
        S = 8
    elif mode == "nondiv":
        S = 9  # 9 % 2 != 0
    else:
        # diff_len: simulate dataloader giving different packed lengths
        S = 8 if rank == 0 else 6

    print(
        f"[rank {rank}] MODE={mode} S={S} world_size={world_size} "
        f"S_mod_world={S % world_size}"
    )

    tokens = torch.randn(1, S, H, device=device)
    # Only broadcast when lengths match; diff_len cannot broadcast full tensor.
    if mode != "diff_len":
        dist.broadcast(tokens, src=0)

    try:
        with context_parallel(mesh, buffers=[tokens], buffer_seq_dims=[1]):
            print(
                f"[rank {rank}] inside CP tokens={tuple(tokens.shape)} "
                f"sum={tokens.float().sum().item():.4f}"
            )
        print(f"[rank {rank}] after CP tokens={tuple(tokens.shape)}")
    except Exception as e:
        print(f"[rank {rank}] FAILED: {type(e).__name__}: {e}")
        # Peer may hang in a collective if only one rank errors — expected for diff_len.

    try:
        dist.barrier()
        if rank == 0:
            print(f"OK: seq-len demo finished (MODE={mode})")
    except Exception as e:
        print(f"[rank {rank}] barrier FAILED: {type(e).__name__}: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
