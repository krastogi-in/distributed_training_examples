"""
CP unshard-while-peer-in-context probe. MODE=healthy|diverge

  What:
    context_parallel shards buffers on enter and (usually) restores on exit.
    context_parallel_unshard() gathers shards back to full sequence — a
    collective that EVERY rank must enter together.

    diverge: rank0 stays inside the `with context_parallel` block (e.g. slow
    work / sleep) while rank1 has exited and calls context_parallel_unshard.
    → collective mismatch / hang.

    healthy: both exit the context, then both call unshard together.

  MODE=diverge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_unshard_while_peer_in_context.py
"""

from __future__ import annotations

import os
import time

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
        with context_parallel(
            mesh,
            buffers=[tokens],
            buffer_seq_dims=[1],
            no_restore_buffers={tokens},
        ):
            print(f"[rank {rank}] inside CP {tuple(tokens.shape)}")
        print(f"[rank {rank}] exited CP, calling unshard together")
        full = context_parallel_unshard(mesh, [tokens], [1])[0]
        print(f"[rank {rank}] after unshard {tuple(full.shape)}")
    else:
        # diverge: both enter; rank0 lingers inside; rank1 exits and unshards
        with context_parallel(
            mesh,
            buffers=[tokens],
            buffer_seq_dims=[1],
            no_restore_buffers={tokens},
        ):
            print(f"[rank {rank}] inside CP {tuple(tokens.shape)}")
            if rank == 0:
                print(f"[rank {rank}] staying in CP context (sleep)…")
                time.sleep(30)
                print(f"[rank {rank}] leaving CP context late")
            else:
                print(f"[rank {rank}] exiting CP soon, then unshard")

        if rank == 1:
            print(f"[rank {rank}] calling context_parallel_unshard while peer may still be in CP")
            try:
                full = context_parallel_unshard(mesh, [tokens], [1])[0]
                print(f"[rank {rank}] unshard returned {tuple(full.shape)}")
            except Exception as e:
                print(f"[rank {rank}] unshard FAILED: {type(e).__name__}: {e}")

    try:
        dist.barrier()
        if rank == 0:
            print(f"OK: unshard-vs-context demo finished (MODE={mode})")
    except Exception as e:
        print(f"[rank {rank}] barrier: {type(e).__name__}: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
