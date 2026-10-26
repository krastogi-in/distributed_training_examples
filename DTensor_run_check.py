"""
DTensor: tensor_meta mismatch across ranks (CPU + gloo).

Build with DTensor.from_local + agreed Shard(0), but diverge meta on rank 1.
Set the same META and FOLLOWUP in both terminals.

  META:     shape | dtype | stride
  FOLLOWUP: none | unshard

Two terminals (2 "nodes", 1 proc each), same machine:

  # terminal A
  META=shape FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_meta_mismatch_ranks.py

  # terminal B
  META=shape FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_meta_mismatch_ranks.py

What to record yourself:
  - global shape / stride / dtype (and _spec.tensor_meta if present) on each rank
  - local shape / stride / dtype / values
  - what happens on FOLLOWUP=unshard (redistribute / full_tensor)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard


def _print_meta(rank: int, tag: str, dt: DTensor) -> None:
    local = dt.to_local()
    print(
        f"[rank {rank}] {tag} "
        f"global_shape={tuple(dt.shape)} global_stride={tuple(dt.stride())} "
        f"dtype={dt.dtype} placements={dt.placements} "
        f"local_shape={tuple(local.shape)} local_stride={tuple(local.stride())} "
        f"local_data={local.flatten().tolist()}"
    )
    spec = getattr(dt, "_spec", None)
    tensor_meta = getattr(spec, "tensor_meta", None) if spec is not None else None
    if tensor_meta is not None:
        print(f"[rank {rank}] {tag} tensor_meta={tensor_meta}")


def _make_local(rank: int, meta: str) -> torch.Tensor:
    """Rank 0 always uses a baseline shard; rank 1 diverges per META."""
    if meta == "shape":
        # Shard(0): global shape = (local_rows * world, cols).
        # Rank0 -> global (4, 2); rank1 -> global (6, 2).
        if rank == 0:
            return torch.arange(4, dtype=torch.float32).reshape(2, 2)
        return torch.arange(6, dtype=torch.float32).reshape(3, 2)

    if meta == "dtype":
        # Same local shape; dtype differs on rank 1.
        if rank == 0:
            return torch.arange(4, dtype=torch.float32).reshape(2, 2)
        return torch.arange(4, dtype=torch.float64).reshape(2, 2)

    if meta == "stride":
        # Same shape/dtype; rank 1 uses a non-row-major stride (transpose view).
        base = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        if rank == 0:
            return base.contiguous()
        return base.t()  # shape (2, 2), stride (1, 2)

    raise SystemExit(f"Unknown META={meta!r}; use shape|dtype|stride")


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    meta = os.environ.get("META", "shape")
    run_check = os.environ.get("RUN_CHECK", "0") == "1"
    followup = os.environ.get("FOLLOWUP", "none")
    dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (world_size,))
    local = _make_local(rank, meta)
    print(
        f"[rank {rank}] META={meta} raw local "
        f"shape={tuple(local.shape)} stride={tuple(local.stride())} dtype={local.dtype}"
    )

    # Same placement argument on both ranks; meta divergence is in the local tensor.
    w = DTensor.from_local(local, mesh, [Shard(0)], run_check=run_check)
    _print_meta(rank, "after from_local", w)

    if followup == "unshard":
        w_rep = w.redistribute(placements=[Replicate()])
        _print_meta(rank, "after redistribute->Replicate", w_rep)
        full = w.full_tensor()
        print(
            f"[rank {rank}] full_tensor shape={tuple(full.shape)} "
            f"stride={tuple(full.stride())} dtype={full.dtype} "
            f"data={full.flatten().tolist()}"
        )
    elif followup != "none":
        raise SystemExit(f"Unknown FOLLOWUP={followup!r}; use none|unshard")

    dist.barrier()
    if rank == 0:
        print(f"OK: meta mismatch demo finished (META={meta}, FOLLOWUP={followup})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
