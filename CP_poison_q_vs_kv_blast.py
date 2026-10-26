"""
CP blast-radius: poison Q (local) vs K/V (shared on ring). MODE=healthy|poison_q|poison_k|poison_v|oom

  What:
    Under CP attention, Q stays local to the rank; K/V shards are rotated /
    exchanged around the ring so every rank sees them. So:
      - poison Q on rank1  → mainly corrupts rank1's output (local blast)
      - poison K or V on rank1 → after exchange, ALL ranks' merges see it (wide blast)

    Also: CP does not enforce a KV memory quota. MODE=oom tries a huge KV
    allocation to show the limit is device memory / OOM, not a CP guardrail.

  CPU is enough for the Q vs K/V blast demo (simulated ring via all_gather).

  MODE=poison_k torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2949 CP_poison_q_vs_kv_blast.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "poison_k")
    assert world_size == 2

    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    B, H, S_local, D = 1, 2, 4, 8

    if mode == "oom":
        # CP has no KV quota — only allocator / device memory stops you.
        # Keep this opt-in; default size is large but may not OOM on big hosts.
        mb = int(os.environ.get("OOM_MB", "512"))
        n = (mb * 1024 * 1024) // 4  # float32
        print(f"[rank {rank}] MODE=oom allocating ~{mb}MB KV-like tensor …")
        try:
            kv = torch.empty(n, device="cpu")
            kv.fill_(1.0)
            print(f"[rank {rank}] alloc OK numel={kv.numel()} (no CP quota blocked us)")
        except Exception as e:
            print(f"[rank {rank}] alloc failed: {type(e).__name__}: {e}")
        dist.barrier()
        if rank == 0:
            print("OK: oom probe finished (limit = OS/allocator, not CP)")
        dist.destroy_process_group()
        return

    q = torch.randn(B, H, S_local, D)
    k = torch.randn(B, H, S_local, D)
    v = torch.randn(B, H, S_local, D)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    # Each rank also has its "local" Q (same broadcast baseline; poison below).
    q_local = q.clone()
    k_local = k.clone()
    v_local = v.clone()

    if mode == "poison_q" and rank == 1:
        q_local = q_local + 50.0
        print(f"[rank {rank}] poisoned LOCAL Q (+50)")
    elif mode == "poison_k" and rank == 1:
        k_local = k_local + 50.0
        print(f"[rank {rank}] poisoned LOCAL K (+50) — will ring to all")
    elif mode == "poison_v" and rank == 1:
        v_local = v_local + 50.0
        print(f"[rank {rank}] poisoned LOCAL V (+50) — will ring to all")

    # Simulate one ring step: all_gather K and V from every rank (like rotate).
    k_list = [torch.empty_like(k_local) for _ in range(world_size)]
    v_list = [torch.empty_like(v_local) for _ in range(world_size)]
    dist.all_gather(k_list, k_local.contiguous())
    dist.all_gather(v_list, v_local.contiguous())
    k_ring = torch.cat(k_list, dim=2)  # full KV seq seen by everyone
    v_ring = torch.cat(v_list, dim=2)

    # Q is NOT gathered — each rank uses only its q_local.
    out = _attn(q_local, k_ring, v_ring)
    print(
        f"[rank {rank}] MODE={mode} out_sum={out.sum().item():.4f} "
        f"out_mean={out.mean().item():.6f}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: Q-vs-KV blast demo finished (MODE={mode})")
        print(
            "Compare: poison_q → ranks differ; poison_k/v → both ranks move together."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
