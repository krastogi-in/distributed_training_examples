"""
DTensor: mesh_dim_names mismatch via named submesh (CPU + gloo).

Same (2, 2) topology on every rank, but dim *names* differ:
  ranks 0,1: ("dp", "tp")
  ranks 2,3: ("tp", "dp")

All ranks then use the *same* named API path: mesh["dp"] + Shard(0).
That is the probe — names select which physical mesh axis / process group.

Needs WORLD_SIZE == 4.

  # terminal A
  torchrun --nnodes=2 --nproc-per-node=2 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_mesh_mismatch_rank.py

  # terminal B
  torchrun --nnodes=2 --nproc-per-node=2 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_mesh_mismatch_rank.py

What to record yourself:
  - mesh_dim_names vs mesh["dp"].mesh (which ranks share the submesh)
  - local shard after distribute on mesh["dp"]
  - whether redistribute / barrier finish on both nodes
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

    if world_size != 4:
        raise SystemExit(
            f"This demo expects WORLD_SIZE=4 for mesh (2, 2); got {world_size}"
        )

    # Intentional name skew only — shape stays (2, 2).
    if rank in (0, 1):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    else:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "dp"))

    # Named lookup: same string on every rank; physical axis depends on local names.
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    print(
        f"[rank {rank}] mesh_dim_names={mesh.mesh_dim_names} "
        f"full_mesh={mesh.mesh.tolist()} "
        f"dp_mesh={dp_mesh.mesh.tolist()} dp_coord={dp_mesh.get_coordinate()} "
        f"tp_mesh={tp_mesh.mesh.tolist()} tp_coord={tp_mesh.get_coordinate()}"
    )

    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    # Agreed code: shard on the submesh named "dp" (1 placement for 1-D submesh).
    w_dp = distribute_tensor(global_w, dp_mesh, [Shard(0)])
    print(
        f"[rank {rank}] after distribute on mesh['dp'] "
        f"local={tuple(w_dp.to_local().shape)} global={tuple(w_dp.shape)} "
        f"placements={w_dp.placements} "
        f"local_data={w_dp.to_local().flatten().tolist()}"
    )

    # Optional gather on that same named mesh — still SPMD in source text.
    w_rep = w_dp.redistribute(placements=[Replicate()])
    print(
        f"[rank {rank}] after redistribute->Replicate on mesh['dp'] "
        f"local={w_rep.to_local().flatten().tolist()}"
    )

    dist.barrier()
    if rank == 0:
        print("OK: named-mesh mismatch demo finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
