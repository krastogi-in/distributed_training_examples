"""
CP rotate-method / load-balance probe. MODE=healthy|rotate|load_balance
Needs CUDA (ring SDPA).

  MODE=rotate torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_rotate_method_diverge.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._context_parallel import (
    set_rotate_method,
)
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _cp_options,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "rotate")

    if not torch.cuda.is_available():
        raise RuntimeError("CP rotate probe needs CUDA")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    mesh = init_device_mesh("cuda", (world_size,))

    # Rotate / load-balance must match across ranks (global _cp_options).
    if mode == "healthy":
        set_rotate_method("allgather")
        _cp_options.enable_load_balance = True
    elif mode == "rotate":
        set_rotate_method("allgather" if rank == 0 else "alltoall")
        _cp_options.enable_load_balance = True
    else:
        set_rotate_method("allgather")
        _cp_options.enable_load_balance = rank == 0

    print(
        f"[rank {rank}] MODE={mode} rotate={_cp_options.rotate_method} "
        f"load_balance={_cp_options.enable_load_balance}"
    )

    torch.manual_seed(0)
    B, S, H = 1, 8, 16
    assert S % world_size == 0
    tokens = torch.randn(B, S, H, device=device)
    dist.broadcast(tokens, src=0)

    with context_parallel(mesh, buffers=[tokens], buffer_seq_dims=[1]):
        t = tokens
        heads, dim = 2, H // 2
        q = t.view(B, t.size(1), heads, dim).transpose(1, 2)
        k = q.clone()
        v = q.clone()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(
            f"[rank {rank}] SDPA out={tuple(out.shape)} "
            f"sum={out.float().sum().item():.4f}"
        )

    dist.barrier()
    if rank == 0:
        print(f"OK: CP rotate demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
