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
from torch.distributed.tensor import Replicate, Partial, DTensor


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (world_size,))
    print(f"[rank {rank}] init world_size={world_size} mesh.shape={tuple(mesh.shape)}")

    local = torch.arange(world_size)
    print(f"[rank {rank}] raw local before from_local: {local.tolist()}")

    if rank == 0:
        tensor = DTensor.from_local(local, mesh, [Partial("max")])

        print(
            f"[rank {rank}] after from_local "
            f"global={tuple(tensor.shape)} placements={tensor.placements} "
            f"local={tensor.to_local().tolist()}"
        )

    else:
        # tensor = DTensor.from_local(torch.tensor([[9,9]]), mesh, [Partial("sum")])
        # tensor = DTensor.from_local(torch.tensor([[9,9]]), mesh, [Partial("avg")])
        tensor = DTensor.from_local(torch.tensor([9,9]), mesh, [Partial("sum")])

        print(
            f"[rank {rank}] after from_local "
            f"global={tuple(tensor.shape)} placements={tensor.placements} "
            f"local={tensor.to_local().tolist()}"
        )

    print(f"[rank {rank}] calling redistribute -> Replicate")
    tensor = tensor.redistribute(placements=[Replicate()])
    print(
        f"[rank {rank}] after Partial->Replicate "
        f"global={tuple(tensor.shape)} placements={tensor.placements} "
        f"local={tensor.to_local().tolist()}"
    )

    dist.barrier()
    if rank == 0:
        print("OK: Partial redistribute demo finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
