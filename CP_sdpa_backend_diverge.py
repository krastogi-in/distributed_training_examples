"""
CP SDPA backend pin diverge. MODE=healthy|diverge

  What:
    SDPA can run flash / mem-efficient / math. Under CP, the ring hooks into
    specific CUDA SDPA backends. If rank0 pins FLASH and rank1 pins MATH, they
    take different kernels/collective paths → usually a LOUD error or hang
    (not silent wrong). Confirm by running diverge vs healthy.

  Needs CUDA.

  MODE=diverge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_sdpa_backend_diverge.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "diverge")

    if not torch.cuda.is_available():
        raise RuntimeError("SDPA backend probe needs CUDA")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    mesh = init_device_mesh("cuda", (world_size,))

    if mode == "healthy":
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        # Prefer flash, allow math fallback — SAME on all ranks
        pin = backends
        print(f"[rank {rank}] MODE=healthy backends={[b.name for b in pin]}")
    else:
        # diverge: rank0 flash-only, rank1 math-only
        pin = [SDPBackend.FLASH_ATTENTION] if rank == 0 else [SDPBackend.MATH]
        print(f"[rank {rank}] MODE=diverge pin={pin[0].name}")

    torch.manual_seed(0)
    B, S, H = 1, 8, 16
    assert S % world_size == 0
    tokens = torch.randn(B, S, H, device=device, dtype=torch.float16)
    dist.broadcast(tokens, src=0)

    try:
        with context_parallel(mesh, buffers=[tokens], buffer_seq_dims=[1]):
            t = tokens
            heads, dim = 2, H // 2
            q = t.view(B, t.size(1), heads, dim).transpose(1, 2)
            k = q.clone()
            v = q.clone()
            with sdpa_kernel(pin):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(
                f"[rank {rank}] SDPA out={tuple(out.shape)} "
                f"sum={out.float().sum().item():.4f}"
            )
        print(f"[rank {rank}] finished cleanly")
    except Exception as e:
        print(f"[rank {rank}] LOUD failure: {type(e).__name__}: {e}")

    try:
        dist.barrier()
        if rank == 0:
            print(f"OK: SDPA backend demo finished (MODE={mode})")
    except Exception as e:
        print(f"[rank {rank}] barrier: {type(e).__name__}: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
