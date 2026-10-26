"""
Hands-on 01: DTensor basics (CPU + gloo).

  torchrun --standalone --nnodes=1 --nproc-per-node=2 DTensor_basic.py

What to watch:
  - each rank's local shard shape
  - placements before/after redistribute
  - Partial -> Replicate uses all-reduce (full_tensor / redistribute)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")

    # 1-D mesh over all ranks (CPU)
    mesh = init_device_mesh("cpu", (world_size,))

    # ----- construct from a "global" tensor -----
    # Only rank 0's values matter for distribute_tensor with default src;
    # others still call it (SPMD).
    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    w_sharded = distribute_tensor(global_w, mesh, [Shard(0)])

    print(
        f"[rank {rank}] Shard(0) local={tuple(w_sharded.to_local().shape)} "
        f"global={tuple(w_sharded.shape)} placements={w_sharded.placements} "
        f"local_data={w_sharded.to_local().flatten().tolist()}"
    )

    # ----- redistribute: Shard -> Replicate (all-gather) -----
    w_rep = w_sharded.redistribute(placements=[Replicate()])
    print(
        f"[rank {rank}] after Shard->Replicate local={w_rep.to_local().flatten().tolist()}"
    )

    # ----- from_local + Partial story -----
    # Pretend each rank holds a partial contribution of a length-2 vector.
    local_partial = torch.zeros(2)
    local_partial[rank % 2] = float(rank + 1)
    p = DTensor.from_local(local_partial, mesh, [Partial()])
    print(
        f"[rank {rank}] Partial local={p.to_local().tolist()} placements={p.placements}"
    )
    p_full = p.redistribute(placements=[Replicate()])
    print(
        f"[rank {rank}] Partial->Replicate (all-reduce) local={p_full.to_local().tolist()}"
    )

    dist.barrier()
    if rank == 0:
        print("OK: hands_on_01 finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
