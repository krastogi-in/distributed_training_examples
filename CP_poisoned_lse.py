"""
Poisoned LSE (log-sum-exp) probe. MODE=healthy|huge|neginf

  What is LSE?
    Softmax needs a stable normalizer. Attention often tracks
      LSE = log(sum_j exp(score_j))
    together with a weighted output. Under CP, each ring step produces a
    partial (out, lse); ranks/chunks MERGE them (online softmax).

  Merge idea (two partials A and B):
      out = out_a - sigmoid(lse_b - lse_a) * (out_a - out_b)
      lse = lse_a - logsigmoid(lse_a - lse_b)
    If one partial's lse is +huge or -inf, the merge is dominated / wiped.

  This script simulates two CP chunks merging (CPU). Rank 1 injects a poisoned
  LSE into its partial before merge — same failure class as a buggy/malicious
  rank in real CP attention.

  MODE=huge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_poisoned_lse.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _partial_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk: returns (out, lse) like a CP sub-SDPA block."""
    # scores: [B, H, S_q, S_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # [B,H,S_q,1]
    p = torch.exp(scores - lse)
    out = torch.matmul(p, v)
    return out, lse


def _merge(out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor):
    """Same structure as CP _SDPAMerger._merge_one (simplified)."""
    out = out_a - F.sigmoid(lse_b - lse_a) * (out_a - out_b)
    lse = lse_a - F.logsigmoid(lse_a - lse_b)
    return out, lse


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "huge")
    assert world_size == 2, "this demo expects 2 ranks (two partials)"

    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    B, H, S, D = 1, 2, 4, 8
    # Shared Q; each rank owns a KV chunk (like ring step / CP shard).
    q = torch.randn(B, H, S, D)
    dist.broadcast(q, src=0)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    # Split KV as two chunks along seq (chunk0 on rank0, chunk1 on rank1).
    k0, k1 = k[..., : S // 2, :], k[..., S // 2 :, :]
    v0, v1 = v[..., : S // 2, :], v[..., S // 2 :, :]

    out0, lse0 = _partial_attn(q, k0, v0)
    out1, lse1 = _partial_attn(q, k1, v1)

    # Exchange partials so both ranks can merge (and see poison effect).
    # Send rank1's (out1,lse1) to rank0 and vice versa via broadcast from each.
    payload_out = out0 if rank == 0 else out1
    payload_lse = lse0 if rank == 0 else lse1

    if mode == "huge" and rank == 1:
        payload_lse = payload_lse + 50.0  # enormous LSE
        print(f"[rank {rank}] poisoned LSE += 50")
    elif mode == "neginf" and rank == 1:
        payload_lse = torch.full_like(payload_lse, float("-inf"))
        print(f"[rank {rank}] poisoned LSE = -inf")

    # Gather both partials onto every rank
    outs = [torch.empty_like(payload_out) for _ in range(world_size)]
    lses = [torch.empty_like(payload_lse) for _ in range(world_size)]
    dist.all_gather(outs, payload_out.contiguous())
    dist.all_gather(lses, payload_lse.contiguous())

    merged, merged_lse = _merge(outs[0], lses[0], outs[1], lses[1])
    print(
        f"[rank {rank}] MODE={mode} "
        f"lse0_mean={lses[0].mean().item():.4f} lse1_mean={lses[1].mean().item():.4f} "
        f"merged_out_sum={merged.sum().item():.4f} merged_lse_mean={merged_lse.mean().item():.4f}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: poisoned-LSE demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
