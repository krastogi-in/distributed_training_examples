"""
CP no_restore_buffers diverge probe. MODE=healthy|diverge

  What:
    context_parallel normally shards buffers on entry and RESTORES full
    sequence on exit. no_restore_buffers={t} means that tensor STAYS sharded
    after the with-block.

    healthy: both ranks pass the same no_restore set (both keep shard).
    diverge: rank 0 restores (default); rank 1 puts tokens in no_restore
             → after exit, shapes disagree (full vs shard).

  MODE=diverge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_no_restore_diverge.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._context_parallel import (
    context_parallel_unshard,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "diverge")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    mesh = init_device_mesh(device, (world_size,))
    torch.manual_seed(0)

    B, S, H = 1, 8, 16
    assert S % world_size == 0
    tokens = torch.randn(B, S, H, device=device)
    dist.broadcast(tokens, src=0)

    print(f"[rank {rank}] MODE={mode} before={tuple(tokens.shape)}")

    if mode == "healthy":
        # Both keep the shard after exit.
        no_restore = {tokens}
    elif rank == 1:
        no_restore = {tokens}
    else:
        no_restore = None  # restore full sequence on exit

    print(f"[rank {rank}] no_restore_buffers={'tokens' if no_restore else 'None'}")

    with context_parallel(
        mesh,
        buffers=[tokens],
        buffer_seq_dims=[1],
        no_restore_buffers=no_restore,
    ):
        print(f"[rank {rank}] inside CP tokens={tuple(tokens.shape)}")

    print(
        f"[rank {rank}] after CP tokens={tuple(tokens.shape)} "
        f"sum={tokens.float().sum().item():.4f}"
    )

    # Only when EVERY rank kept the shard (healthy): gather back together.
    if mode == "healthy":
        full = context_parallel_unshard(mesh, [tokens], [1])[0]
        print(f"[rank {rank}] after explicit unshard={tuple(full.shape)}")

    dist.barrier()
    if rank == 0:
        print(f"OK: CP no_restore demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
