"""
Hands-on 04: Context Parallel — sequence shard + (if CUDA) attention under CP.

  # Prefer 2+ GPUs:
  torchrun --standalone --nnodes=1 --nproc-per-node=2 hands_on_04_cp.py

On CPU-only machines this script still shows sequence sharding of buffers;
ring SDPA path needs CUDA.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._context_parallel import (
    context_parallel_unshard,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)

    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        if rank == 0:
            print(
                "WARN: no CUDA multi-GPU — demoing sequence buffer shard only; "
                "ring attention needs GPU."
            )

    mesh = init_device_mesh(device_type, (world_size,))

    # Fake token batch: [B, S, H] — shard along sequence dim 1
    torch.manual_seed(0)
    B, S, H = 1, 8, 16
    assert S % world_size == 0
    tokens = torch.randn(B, S, H, device=device)
    dist.broadcast(tokens.cpu() if tokens.device.type != "cpu" else tokens, src=0)
    if use_cuda:
        tokens = tokens.cuda()

    print(f"[rank {rank}] before CP tokens={tuple(tokens.shape)}")

    with context_parallel(
        mesh,
        buffers=[tokens],
        buffer_seq_dims=[1],
    ):
        print(f"[rank {rank}] inside CP tokens={tuple(tokens.shape)}")

        if use_cuda:
            # SDPA under CP dispatcher (ring). Shapes: [B, heads, S_local, dim]
            # Build QKV from local token shard for a tiny smoke test.
            t = tokens  # [B, S/world, H]
            # reshape to attention layout
            heads, dim = 2, H // 2
            q = t.view(B, t.size(1), heads, dim).transpose(1, 2)
            k = q.clone()
            v = q.clone()
            # seq dim for SDPA is 2
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"[rank {rank}] SDPA out={tuple(out.shape)}")
        else:
            # CPU: just prove local shard is smaller on seq dim
            local_sum = tokens.sum()
            print(f"[rank {rank}] local token sum={local_sum.item():.4f}")

    print(f"[rank {rank}] after CP (restored) tokens={tuple(tokens.shape)}")

    # Demonstrate unshard: keep a sharded buffer (no restore), then gather
    sharded = tokens.clone()
    with context_parallel(
        mesh,
        buffers=[sharded],
        buffer_seq_dims=[1],
        no_restore_buffers={sharded},
    ):
        print(f"[rank {rank}] no_restore sharded tokens={tuple(sharded.shape)}")
    full = context_parallel_unshard(mesh, [sharded], [1])[0]
    print(f"[rank {rank}] after context_parallel_unshard={tuple(full.shape)}")

    dist.barrier()
    if rank == 0:
        print("OK: hands_on_04 CP finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
