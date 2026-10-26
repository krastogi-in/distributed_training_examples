"""
DTensor: SPMD strategy / OpSchema / cost-model skew on one rank (CPU + gloo).

Umbrella (your backlog j): ranks must share the same sharding rules.
This script intentionally breaks that on rank 1 only — distinct from
placement (n) and tensor_meta (m) probes.

  MODE:
    missing_strategy  — drop OpSchema strategy for aten.add.Tensor on rank 1 (o / a-like)
    cost_flip         — rank 1 picks max redistribute cost instead of min (k / l)
    decomp_gap        — rank 1 drops aten.gelu strategy (forces different prop/decomp path)

  FOLLOWUP is unused; the op itself is the probe.

Two terminals (same MODE in both):

  # terminal A
  MODE=missing_strategy torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_spmd_strategy_mismatch_ranks.py

  # terminal B
  MODE=missing_strategy torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_spmd_strategy_mismatch_ranks.py

What to record yourself:
  - whether both ranks apply the same local patch
  - whether add / mm / gelu completes, errors, or stalls
  - local results if both sides print
"""

from __future__ import annotations

import os
from itertools import chain

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def _propagator():
    return DTensor._op_dispatcher.sharding_propagator


def _clear_prop_caches(prop) -> None:
    for name in ("propagate_op_sharding", "_propagate_tensor_meta", "propagate"):
        fn = getattr(prop, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def _patch_rank1(mode: str, rank: int) -> str:
    """Apply intentional SPMD rule skew on rank 1 only. Returns a short note."""
    if rank != 1:
        return "no patch (rank 0 keeps stock strategies/cost)"

    prop = _propagator()
    _clear_prop_caches(prop)

    if mode == "missing_strategy":
        op = torch.ops.aten.add.Tensor
        removed = prop.op_strategy_funcs.pop(op, None)
        _clear_prop_caches(prop)
        return f"removed op_strategy_funcs[{op}] ({'hit' if removed else 'miss'})"

    if mode == "cost_flip":
        orig = prop._select_strategy

        def _select_max_cost(strategy):
            if len(strategy.strategies) == 1:
                return strategy.strategies[0]
            costs = []
            for op_spec in strategy.strategies:
                assert op_spec.redistribute_cost is not None
                costs.append(sum(chain.from_iterable(op_spec.redistribute_cost)))
            # stock code uses min(...); this rank intentionally flips the pick
            return strategy.strategies[costs.index(max(costs))]

        prop._select_strategy = _select_max_cost  # type: ignore[method-assign]
        _clear_prop_caches(prop)
        return "patched _select_strategy -> max redistribute cost"

    if mode == "decomp_gap":
        op = torch.ops.aten.gelu.default
        removed = prop.op_strategy_funcs.pop(op, None)
        _clear_prop_caches(prop)
        return f"removed op_strategy_funcs[{op}] ({'hit' if removed else 'miss'})"

    raise SystemExit(f"Unknown MODE={mode!r}; use missing_strategy|cost_flip|decomp_gap")


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "missing_strategy")
    dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (world_size,))
    note = _patch_rank1(mode, rank)
    print(f"[rank {rank}] MODE={mode} patch={note}")

    # Agreed layout (not an n/m probe): both ranks same global + Shard(0).
    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    x = distribute_tensor(global_w, mesh, [Shard(0)])
    y = distribute_tensor(global_w + 1, mesh, [Shard(0)])
    print(
        f"[rank {rank}] inputs ready "
        f"x.local={tuple(x.to_local().shape)} placements={x.placements}"
    )

    # Barrier so both ranks enter the op with patches already applied.
    dist.barrier()

    if mode in ("missing_strategy", "cost_flip"):
        # pointwise / broadcast-style prop uses add strategies + cost ranking
        out = x + y
        print(
            f"[rank {rank}] after add "
            f"local={out.to_local().flatten().tolist()} placements={out.placements}"
        )
        # mm often has multiple redistribute strategies -> better for cost_flip
        a = distribute_tensor(
            torch.arange(8, dtype=torch.float32).reshape(4, 2), mesh, [Shard(0)]
        )
        b = distribute_tensor(
            torch.arange(8, dtype=torch.float32).reshape(2, 4), mesh, [Shard(1)]
        )
        prod = a.reshape(4, 2) @ b.reshape(2, 4)
        print(
            f"[rank {rank}] after mm "
            f"local_shape={tuple(prod.to_local().shape)} "
            f"placements={prod.placements} "
            f"local={prod.to_local().flatten().tolist()}"
        )

    elif mode == "decomp_gap":
        out = torch.nn.functional.gelu(x)
        print(
            f"[rank {rank}] after gelu "
            f"local={out.to_local().flatten().tolist()} placements={out.placements}"
        )

    dist.barrier()
    if rank == 0:
        print(f"OK: SPMD strategy mismatch demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
