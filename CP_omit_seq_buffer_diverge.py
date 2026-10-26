"""
CP omit seq-dependent buffer probe. MODE=healthy|omit
Works on CPU (buffer shard) or CUDA.

  MODE=omit torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_omit_seq_buffer_diverge.py
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
    mode = os.environ.get("MODE", "omit")
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
    # Stand-in for RoPE freqs_cis: sequence-dependent, must be CP-sharded too.
    freqs_cis = torch.arange(S, device=device, dtype=torch.float32).view(1, S, 1)
    freqs_cis = freqs_cis.expand(B, S, H).contiguous()

    dist.broadcast(tokens, src=0)
    dist.broadcast(freqs_cis, src=0)

    print(
        f"[rank {rank}] MODE={mode} before CP "
        f"tokens={tuple(tokens.shape)} freqs={tuple(freqs_cis.shape)}"
    )

    if mode == "healthy" or rank == 0:
        buffers = [tokens, freqs_cis]
        seq_dims = [1, 1]
        print(f"[rank {rank}] buffers=[tokens, freqs_cis]")
    else:
        buffers = [tokens]
        seq_dims = [1]
        print(f"[rank {rank}] buffers=[tokens]  (freqs_cis omitted)")

    with context_parallel(mesh, buffers=buffers, buffer_seq_dims=seq_dims):
        print(
            f"[rank {rank}] inside CP tokens={tuple(tokens.shape)} "
            f"freqs={tuple(freqs_cis.shape)}"
        )
        # Combine like a RoPE-style use of freqs along the seq dim.
        out = tokens * freqs_cis
        print(
            f"[rank {rank}] out.shape={tuple(out.shape)} "
            f"out.sum={out.float().sum().item():.4f}"
        )

    print(
        f"[rank {rank}] after CP tokens={tuple(tokens.shape)} "
        f"freqs={tuple(freqs_cis.shape)}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: CP omit-buffer demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
