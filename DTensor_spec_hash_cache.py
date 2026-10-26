"""
DTensorSpec hash vs eq → sharding-prop cache confusion (CPU + gloo).

MODE:
  baseline       — print hash/eq for two different specs; warm prop cache (no poison)
  stale_hash     — rank 1 only: change tensor_meta while freezing stale _hash
                   (breaks hash/eq contract: same cached hash, different eq key material)
  collide_hash   — rank 1 only: force spec_B._hash = hash(spec_A) while A != B

  # terminal A
  MODE=stale_hash torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_spec_hash_cache.py

  # terminal B
  MODE=stale_hash torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_spec_hash_cache.py

What to record yourself:
  - hash(a)/hash(b), a==b before and after poison
  - propagate_op_sharding cache_info hits/misses
  - whether both ranks finish the same add / print same local sums
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta


def _cache_info():
    prop = DTensor._op_dispatcher.sharding_propagator
    fn = prop.propagate_op_sharding
    if hasattr(fn, "cache_info"):
        return fn.cache_info()
    # LocalLRUCache wrapper
    if hasattr(fn, "cache") and hasattr(fn.cache, "cache_info"):
        return fn.cache.cache_info()
    return None


def _clear_cache() -> None:
    prop = DTensor._op_dispatcher.sharding_propagator
    for name in ("propagate_op_sharding", "_propagate_tensor_meta"):
        fn = getattr(prop, name, None)
        if fn is None:
            continue
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
        elif hasattr(fn, "cache") and hasattr(fn.cache, "cache_clear"):
            fn.cache.cache_clear()


def _log_spec(rank: int, tag: str, spec: DTensorSpec) -> None:
    print(
        f"[rank {rank}] {tag} placements={spec.placements} "
        f"shape={None if spec.tensor_meta is None else tuple(spec.tensor_meta.shape)} "
        f"hash={hash(spec)} _hash_field={getattr(spec, '_hash', None)}"
    )


def _poison_stale_hash(spec: DTensorSpec) -> None:
    """Keep old _hash, replace tensor_meta (bypass DTensorSpec.__setattr__)."""
    # Force hash to be materialized first.
    _ = hash(spec)
    old_hash = spec._hash
    new_meta = TensorMeta(
        shape=torch.Size([8, 2]),  # lied global shape
        stride=(2, 1),
        dtype=spec.tensor_meta.dtype,  # type: ignore[union-attr]
    )
    # Bypass __setattr__ so _hash is NOT cleared.
    object.__setattr__(spec, "tensor_meta", new_meta)
    object.__setattr__(spec, "_hash", old_hash)
    print(
        f"  poisoned stale_hash: kept _hash={old_hash} "
        f"but shape now={tuple(new_meta.shape)}; "
        f"hash()={hash(spec)} recomputed_key_hash={spec._hash_impl()}"
    )


def _poison_collide_hash(spec_a: DTensorSpec, spec_b: DTensorSpec) -> None:
    """Make hash(spec_b) report hash(spec_a) while a != b."""
    _ = hash(spec_a)
    _ = hash(spec_b)
    object.__setattr__(spec_b, "_hash", spec_a._hash)
    print(
        f"  poisoned collide: hash(a)={hash(spec_a)} hash(b)={hash(spec_b)} "
        f"a==b={spec_a == spec_b} b._hash_impl()={spec_b._hash_impl()}"
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "baseline")
    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (world_size,))
    _clear_cache()

    g0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    g1 = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    a = distribute_tensor(g0, mesh, [Shard(0)])
    b = distribute_tensor(g1, mesh, [Shard(0)])

    spec_a = a._spec
    spec_b = b._spec
    _log_spec(rank, "spec_a", spec_a)
    _log_spec(rank, "spec_b", spec_b)
    print(
        f"[rank {rank}] MODE={mode} a==b={spec_a == spec_b} "
        f"hash_equal={hash(spec_a) == hash(spec_b)}"
    )

    if mode == "stale_hash" and rank == 1:
        _poison_stale_hash(spec_a)
        _log_spec(rank, "spec_a after poison", spec_a)
        print(
            f"[rank {rank}] after poison: hash(a)==_hash_impl? "
            f"{hash(spec_a) == spec_a._hash_impl()} a==b={spec_a == spec_b}"
        )
    elif mode == "collide_hash" and rank == 1:
        _poison_collide_hash(spec_a, spec_b)
        _log_spec(rank, "spec_b after poison", spec_b)
    elif mode not in ("baseline", "stale_hash", "collide_hash"):
        raise SystemExit(f"Unknown MODE={mode!r}; use baseline|stale_hash|collide_hash")

    dist.barrier()

    # Warm / exercise sharding-prop cache with a simple DTensor op.
    info0 = _cache_info()
    print(f"[rank {rank}] cache before ops: {info0}")

    # Two adds — second may hit cache if schema+specs match.
    out1 = a + a
    info1 = _cache_info()
    out2 = a + a
    info2 = _cache_info()
    out3 = b + b
    info3 = _cache_info()

    print(
        f"[rank {rank}] cache after a+a: {info1}; after 2nd a+a: {info2}; after b+b: {info3}"
    )
    print(
        f"[rank {rank}] out1.local_sum={out1.to_local().sum().item():.4f} "
        f"out2.local_sum={out2.to_local().sum().item():.4f} "
        f"out3.local_sum={out3.to_local().sum().item():.4f} "
        f"a.placements={a.placements} a.shape={tuple(a.shape)}"
    )

    dist.barrier()
    if rank == 0:
        print(f"OK: spec-hash cache demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
