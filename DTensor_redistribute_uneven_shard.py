"""
DTensor: redistribute the same logical tensor to different placements per rank
(CPU + gloo).

Two terminals (2 "nodes", 1 proc each), same machine:

  # terminal A
  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_redistribute_shard_mismatch_ranks.py

  # terminal B
  torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_redistribute_shard_mismatch_ranks.py

Scenario:
  Start agreed on Replicate; then
  rank 0: redistribute([Shard(0)]), peer: redistribute([Shard(1)]).

What to record yourself:
  - local shapes / placements / local_data after redistribute
  - whether both ranks reach the final prints / barrier
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (world_size,))

    global_w = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    w = distribute_tensor(global_w, mesh, [Replicate()])
    print(
        f"[rank {rank}] start Replicate local={tuple(w.to_local().shape)} "
        f"global={tuple(w.shape)} placements={w.placements}"
    )

    target = [Shard(0)] if rank == 0 else [Shard(0)]
    print(f"[rank {rank}] redistribute target={target}")
    w_sharded = w.redistribute(placements=target)
    print(
        f"[rank {rank}] after redistribute "
        f"local={tuple(w_sharded.to_local().shape)} "
        f"global={tuple(w_sharded.shape)} placements={w_sharded.placements} "
        f"local_data={w_sharded.to_local().flatten().tolist()}"
    )

    dist.barrier()
    if rank == 0:
        print("OK: redistribute placement-differ demo finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
