"""
Causal / position-index breakage probe. MODE=healthy|permute

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
    q = tokens.view(B, S, heads, dim).transpose(1, 2)
    k = q.clone()
    v = q.clone()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).contiguous().view(B, S, H)


def _leakage(tokens: torch.Tensor) -> tuple[torch.Tensor, float]:
    """How much position-0 output moves when the last-slot spike is removed."""
    out = _sdpa(tokens)
    base = tokens.clone()
    base[:, -1, :] = 0.0
    base_out = _sdpa(base)
    leak = (out[:, 0] - base_out[:, 0]).abs().mean().item()
    return out, leak


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "permute")
    dist.init_process_group(backend="gloo")
    torch.manual_seed(0)

    B, S, H = 1, 8, 16
    assert S % world_size == 0
    S_local = S // world_size

    tokens = torch.zeros(B, S, H)
    tokens[0, -1, :] = 100.0  # future spike
    dist.broadcast(tokens, src=0)

    # --- single-device reference ---
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
        base[:, -1, :] = 0.0
        out_nc_base = F.scaled_dot_product_attention(
            base.view(B, S, 2, H // 2).transpose(1, 2),
            base.view(B, S, 2, H // 2).transpose(1, 2),
            base.view(B, S, 2, H // 2).transpose(1, 2),
            is_causal=False,
        )
        leak_nc = (out_nc.transpose(1, 2)[:, 0] - out_nc_base.transpose(1, 2)[:, 0]).abs().mean().item()
        print(
            f"[rank 0] REF causal leakage_metric={leak_ref:.6f} (~0) "
            f"noncausal leakage_metric={leak_nc:.6f} (>>0)"
        )

    start = rank * S_local
    local = tokens[:, start : start + S_local, :].contiguous()
    print(f"[rank {rank}] MODE={mode} shard=[{start}:{start + S_local}]")

    if mode == "permute" and rank == 1:
        # Move last local token (global future spike on last rank) to index 0.
        perm = torch.arange(S_local)
        perm[0], perm[-1] = perm[-1].item(), perm[0].item()
        local = local[:, perm, :].contiguous()
        print(f"[rank {rank}] PERMUTED local order perm={perm.tolist()}")

    out, leak = _leakage(local)
    print(
        f"[rank {rank}] out_pos0_mean={out[0, 0].mean().item():.6f} "
        f"out_sum={out.sum().item():.4f} leakage_metric={leak:.6f}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: causal/index-breakage demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
