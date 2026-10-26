"""
Packed-sequence document-mask isolation probe. MODE=healthy|nomask|broken_span

  Why nomask looked fine before:
    Spike was in doc1 (later). Causal already stops doc0 from seeing it.
    Document isolation matters for the OTHER direction: doc1 must not see doc0.

  Fix: spike in doc0; measure effect on doc1 outputs.
    healthy  = same-doc & causal     → doc1_leakage ~ 0
    nomask   = causal only           → doc1_leakage >> 0  (cross-sample!)
    broken_span = doc mask with a hole allowing doc1 → doc0 spike

  MODE=nomask torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_packed_doc_mask_isolation.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _doc_ids_for_packed(S: int, boundaries: list[int]) -> torch.Tensor:
    doc = torch.empty(S, dtype=torch.long)
    for d, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        doc[a:b] = d
    return doc


def _doc_mask(doc_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
    S = doc_ids.numel()
    same = doc_ids.view(S, 1) == doc_ids.view(1, S)
    if causal:
        return same & torch.tril(torch.ones(S, S, dtype=torch.bool))
    return same


def _sdpa(tokens: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
    B, S, H = tokens.shape
    heads, dim = 2, H // 2
    q = tokens.view(B, S, heads, dim).transpose(1, 2)
    k = q.clone()
    v = q.clone()
    float_mask = torch.zeros(S, S, device=tokens.device)
    float_mask = float_mask.masked_fill(~allow, float("-inf"))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
    return out.transpose(1, 2).contiguous().view(B, S, H)


def _leak_doc1_from_doc0_spike(
    tokens: torch.Tensor, allow: torch.Tensor, doc0_end: int, doc1_end: int
) -> float:
    """Spike lives in doc0; watch whether doc1 outputs change when spike removed."""
    out = _sdpa(tokens, allow)
    base = tokens.clone()
    base[:, :doc0_end, :] = 0.0
    base_out = _sdpa(base, allow)
    return (out[:, doc0_end:doc1_end] - base_out[:, doc0_end:doc1_end]).abs().mean().item()


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "healthy")
    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    # doc0=[0:3), doc1=[3:6), doc2=[6:8) — doc1 spans CP boundary at 4 when world=2
    B, S, H = 1, 8, 16
    boundaries = [0, 3, 6, 8]
    doc0_end, doc1_end = 3, 6
    doc_ids = _doc_ids_for_packed(S, boundaries)

    tokens = torch.zeros(B, S, H)
    tokens[0, 1, :] = 100.0  # spike inside doc0 (early)
    dist.broadcast(tokens, src=0)

    allow = _doc_mask(doc_ids, causal=True)
    if mode == "nomask":
        # Causal only — doc1 MAY attend doc0 (cross-sample leak).
        allow = torch.tril(torch.ones(S, S, dtype=torch.bool))
    elif mode == "broken_span":
        allow = allow.clone()
        # Hole: allow a doc1 query to see the doc0 spike key.
        allow[4, 1] = True

    leak = _leak_doc1_from_doc0_spike(tokens, allow, doc0_end, doc1_end)
    print(
        f"[rank {rank}] MODE={mode} doc_ids={doc_ids.tolist()} "
        f"spike_at=1 doc1_leakage={leak:.6f}"
    )

    S_local = S // world_size
    start = rank * S_local
    print(
        f"[rank {rank}] CP-shard[{start}:{start + S_local}] "
        f"local_doc_ids={doc_ids[start:start + S_local].tolist()}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: packed doc-mask demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
