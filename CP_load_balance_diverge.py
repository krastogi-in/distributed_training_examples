"""
CP load-balance probe. MODE=healthy|balance

  What (why this exists):
    Under causal CP attention, later sequence chunks have more past tokens to
    attend to → more work. Load balancing rearranges shards so ranks share that
    work more evenly. If enable_load_balance disagrees across ranks, the ring
    skips/computes different KV chunks → hang or wrong attn (CUDA path).

    "Overutilized rank" = with load_balance OFF, higher ranks typically do more
    causal FLOPs (same schedule on all ranks; just uneven work). Diverging the
    FLAG across ranks is the SPMD bug; uneven work alone is expected.

  Needs CUDA (CP ring SDPA).

  MODE=balance torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_load_balance_diverge.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._context_parallel import set_rotate_method
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _cp_options,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "balance")

    if not torch.cuda.is_available():
        raise RuntimeError("CP load-balance probe needs CUDA")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    mesh = init_device_mesh("cuda", (world_size,))

    set_rotate_method("allgather")

    if mode == "healthy":
        _cp_options.enable_load_balance = True
    else:
        # balance: rank 0 on, rank 1 off
        _cp_options.enable_load_balance = rank == 0

    print(
        f"[rank {rank}] MODE={mode} "
        f"enable_load_balance={_cp_options.enable_load_balance}"
    )

    torch.manual_seed(0)
    B, S, H = 1, 8, 16
    assert S % world_size == 0
    tokens = torch.randn(B, S, H, device=device)
    dist.broadcast(tokens, src=0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

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

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[rank {rank}] wall_time_s={dt:.4f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: CP load-balance demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
