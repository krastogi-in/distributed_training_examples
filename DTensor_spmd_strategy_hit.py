"""
DTensor SPMD strategy skew that actually *hits* on current PyTorch.

The older probe popped op_strategy_funcs[aten.add.Tensor] → (miss).
On this build, pointwise/mm/gelu live in op_single_dim_strategy_funcs.

  MODE:
    missing_add   — rank1 pops aten.add.Tensor from single-dim registry
    missing_gelu  — rank1 pops aten.gelu.default
    missing_mm    — rank1 pops aten.mm.default
    cost_flip     — rank1 selects MAX redistribute cost instead of min

  MODE=missing_add torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=2950 DTensor_spmd_strategy_hit.py
"""

from __future__ import annotations

import os
from itertools import chain

import torch
import torch.distributed as dist
import torch.distributed.tensor._sharding_prop as sharding_prop
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def _prop():
    return DTensor._op_dispatcher.sharding_propagator


def _clear_caches(prop) -> None:
    for name in (
        "propagate_op_sharding",
        "_propagate_tensor_meta",
        "propagate",
        "propagate_op_sharding_non_cached",
    ):
        fn = getattr(prop, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    # lru on module helpers if present
    for name in ("_propagate_tensor_meta_cached",):
        fn = getattr(prop, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def _pop_single(op, rank: int) -> str:
    prop = _prop()
    _clear_caches(prop)
    removed = prop.op_single_dim_strategy_funcs.pop(op, None)
    # also clear legacy dict if present
    prop.op_strategy_funcs.pop(op, None)
    _clear_caches(prop)
    tag = "hit" if removed is not None else "miss"
    return f"pop op_single_dim_strategy_funcs[{op}] ({tag})"


def _patch_cost_flip(rank: int) -> str:
    prop = _prop()
    _clear_caches(prop)
    orig = sharding_prop._select_min_cost_strategy

    def _select_max_cost(strategy, op_schema=None):
        if len(strategy.strategies) == 1:
            return strategy.strategies[0]
        costs = []
        for op_spec in strategy.strategies:
            assert op_spec.redistribute_cost is not None
            costs.append(sum(chain.from_iterable(op_spec.redistribute_cost)))
        return strategy.strategies[costs.index(max(costs))]

    # Runtime global lookup inside propagate_op_sharding_non_cached
    sharding_prop._select_min_cost_strategy = _select_max_cost  # type: ignore[assignment]
    _clear_caches(prop)
    return "patched _select_min_cost_strategy -> max cost"


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "missing_add")
    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (world_size,))

    if rank == 0:
        note = "no patch (stock)"
    elif mode == "missing_add":
        note = _pop_single(torch.ops.aten.add.Tensor, rank)
    elif mode == "missing_gelu":
        note = _pop_single(torch.ops.aten.gelu.default, rank)
    elif mode == "missing_mm":
        note = _pop_single(torch.ops.aten.mm.default, rank)
    elif mode == "cost_flip":
        note = _patch_cost_flip(rank)
    else:
        raise SystemExit(
            f"Unknown MODE={mode!r}; use missing_add|missing_gelu|missing_mm|cost_flip"
        )

    print(f"[rank {rank}] MODE={mode} patch={note}", flush=True)

    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    x = distribute_tensor(global_w, mesh, [Shard(0)])
    y = distribute_tensor(global_w + 1, mesh, [Shard(0)])
    print(
        f"[rank {rank}] inputs ready x.local={tuple(x.to_local().shape)} "
        f"placements={x.placements}",
        flush=True,
    )
    dist.barrier()

    try:
        if mode in ("missing_add", "cost_flip"):
            out = x + y
            print(
                f"[rank {rank}] after add "
                f"local={out.to_local().flatten().tolist()} "
                f"placements={out.placements}",
                flush=True,
            )
            a = distribute_tensor(
                torch.arange(8, dtype=torch.float32).reshape(4, 2), mesh, [Shard(0)]
            )
            b = distribute_tensor(
                torch.arange(8, dtype=torch.float32).reshape(2, 4), mesh, [Shard(1)]
            )
            prod = a @ b
            print(
                f"[rank {rank}] after mm "
                f"local_shape={tuple(prod.to_local().shape)} "
                f"placements={prod.placements} "
                f"local={prod.to_local().flatten().tolist()}",
                flush=True,
            )
        elif mode == "missing_gelu":
            out = torch.nn.functional.gelu(x)
            print(
                f"[rank {rank}] after gelu "
                f"local={out.to_local().flatten().tolist()} "
                f"placements={out.placements}",
                flush=True,
            )
        elif mode == "missing_mm":
            a = distribute_tensor(
                torch.arange(8, dtype=torch.float32).reshape(4, 2), mesh, [Shard(0)]
            )
            b = distribute_tensor(
                torch.arange(8, dtype=torch.float32).reshape(2, 4), mesh, [Shard(1)]
            )
            prod = a @ b
            print(
                f"[rank {rank}] after mm "
                f"local_shape={tuple(prod.to_local().shape)} "
                f"placements={prod.placements} "
                f"local={prod.to_local().flatten().tolist()}",
                flush=True,
            )
    except Exception as e:
        print(f"[rank {rank}] LOUD failure: {type(e).__name__}: {e}", flush=True)
        dist.destroy_process_group()
        raise

    dist.barrier()
    if rank == 0:
        print(f"OK: strategy-hit demo finished (MODE={mode})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
