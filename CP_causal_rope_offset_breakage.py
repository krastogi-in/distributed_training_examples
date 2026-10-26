"""
Causal / RoPE-offset breakage probe. MODE=healthy|permute

  What:
    Causal SDPA masks by TENSOR INDEX (i cannot attend j > i), not by the
    RoPE id values. Shuffling RoPE angles alone barely changes things and
    does NOT create future-token leakage.

    This probe plants a spike on the last token, then on rank 1 PERMUTES the
    token order (indices). Causal still applies to layout indices, so content
    that was "future" can land in earlier slots and leak into early outputs.
    Compare leakage_metric to single-device reference (~0) and to healthy.

  MODE=permute torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 CP_causal_rope_offset_breakage.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _sdpa(tokens: torch.Tensor) -> torch.Tensor:
    B, S, H = tokens.shape
    heads, dim = 2, H // 2
    # RoPE-like: rotate pairs by angle ~ position
    t = tokens.clone()
    half = H // 2
    angles = pos.to(t.dtype) * 0.01  # [S]
    cos = angles.cos().view(1, S, 1)
    sin = angles.sin().view(1, S, 1)
    a, b = t[..., :half], t[..., half:]
    t = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

    q = t.view(B, S, heads, dim).transpose(1, 2)
    k = q.clone()
    v = t.view(B, S, heads, dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).contiguous().view(B, S, H)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "permute")
    backend = "gloo"
    dist.init_process_group(backend=backend)
    torch.manual_seed(0)

    B, S, H = 1, 8, 16
    assert S % world_size == 0
    S_local = S // world_size

    tokens = torch.zeros(B, S, H)
    # Plant future spike only on the last position.
    tokens[0, -1, :] = 100.0
    dist.broadcast(tokens, src=0)

    # --- single-device reference (full sequence, correct positions) ---
    if rank == 0:
        _, leak_ref = _leakage(tokens)
        _, leak_full = _leakage(tokens)  # same
        out_nc = F.scaled_dot_product_attention(
            tokens.view(B, S, 2, H // 2).transpose(1, 2),
            tokens.view(B, S, 2, H // 2).transpose(1, 2),
            tokens.view(B, S, 2, H // 2).transpose(1, 2),
            is_causal=False,
        )
        base = tokens.clone()
        base[0, -1, :] = 0.0
        base_out = _sdpa_from_tokens(base, ref_pos)
        leak_ref = (ref_out[0, 0] - base_out[0, 0]).abs().mean().item()
        print(
            f"[rank 0] REF pos0 mean={ref_out[0, 0].mean().item():.6f} "
            f"leakage_metric={leak_ref:.6f} (should be ~0 for causal)"
        )

    # --- per-rank shard with global position ids ---
    start = rank * S_local
    local_tokens = tokens[:, start : start + S_local, :].contiguous()
    pos = torch.arange(start, start + S_local)

    if mode == "permute" and rank == 1:
        # Break RoPE/position alignment on this rank only.
        pos = pos[torch.randperm(S_local)]
        print(f"[rank {rank}] PERMUTED pos={pos.tolist()}")
    else:
        print(f"[rank {rank}] pos={pos.tolist()}")

    out = _sdpa_from_tokens(local_tokens, pos)
    print(
        f"[rank {rank}] MODE={mode} local_tokens={tuple(local_tokens.shape)} "
        f"out_pos0_mean={out[0, 0].mean().item():.6f} out_sum={out.sum().item():.4f}"
    )

    # If this shard contains the last token (rank world-1), also report
    # how much the FIRST local position moved vs zero-spike local baseline.
    if rank == world_size - 1:
        base_local = local_tokens.clone()
        base_local[0, -1, :] = 0.0
        # Use SAME (possibly wrong) pos to isolate causal+pos interaction.
        base_out = _sdpa_from_tokens(base_local, pos)
        leak = (out[0, 0] - base_out[0, 0]).abs().mean().item()
        print(f"[rank {rank}] local_leakage_metric={leak:.6f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: causal/RoPE-offset demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
