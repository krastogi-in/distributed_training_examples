"""
DTensor: restrict fast path (sharding-prop cache) on one rank only.

Rank 0 warms the prop cache; rank 1 clears it every time (forced slow path).
If strategies still match, math should match — this is a CPU/cache asymmetry,
not an SPMD placement bug.

  # terminal A
  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_fast_path_one_rank.py

  # terminal B
  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_fast_path_one_rank.py

What to record yourself:
  - cache_info hits/misses on each rank
  - local_sum / gathered out equality across ranks
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def _prop():
    return DTensor._op_dispatcher.sharding_propagator


def _cache_info():
    prop = _prop()
    fn = prop.propagate_op_sharding
    if hasattr(fn, "cache_info"):
        return fn.cache_info()
    if hasattr(fn, "cache") and hasattr(fn.cache, "cache_info"):
        return fn.cache.cache_info()
    return None


def _clear_prop_caches() -> None:
    prop = _prop()
    for name in ("propagate_op_sharding", "_propagate_tensor_meta", "propagate"):
        fn = getattr(prop, name, None)
        if fn is None:
            continue
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
        elif hasattr(fn, "cache") and hasattr(fn.cache, "cache_clear"):
            fn.cache.cache_clear()


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (world_size,))

    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    w = distribute_tensor(global_w, mesh, [Shard(0)])

    # Both ranks start clean.
    _clear_prop_caches()
    print(f"[rank {rank}] cache after clear: {_cache_info()}")

    dist.barrier()

    # Rank 0: warm fast path (run twice; second may hit cache).
    # Rank 1: clear before every op → stay on slow path.
    def run_add(tag: str):
        if rank == 1:
            _clear_prop_caches()
        before = _cache_info()
        out = w + w
        after = _cache_info()
        print(
            f"[rank {rank}] {tag} local_sum={out.to_local().sum().item():.4f} "
            f"cache_before={before} cache_after={after}"
        )
        return out

    out1 = run_add("add#1")
    out2 = run_add("add#2")

    # Cross-check: all_gather local sums (should match pattern of shards, not diverge oddly).
    # For equality of full result, unshard and compare.
    full = out2.full_tensor()
    print(f"[rank {rank}] full_tensor sum={full.sum().item():.4f} data={full.flatten().tolist()}")

    # Agree on a single scalar across ranks.
    s = torch.tensor([full.sum().item()], dtype=torch.float64)
    gathered = [torch.zeros_like(s) for _ in range(world_size)]
    dist.all_gather(gathered, s)
    if rank == 0:
        vals = [t.item() for t in gathered]
        print(
            f"[rank 0] full_sum by rank={vals} "
            f"all_equal={all(v == vals[0] for v in vals)}"
        )
        print("OK: fast-path-one-rank demo finished")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
