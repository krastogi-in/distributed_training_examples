"""
DTensor: construct the same logical tensor with different placements per rank
(CPU + gloo), then optionally unshard or all-reduce-average.

Two terminals (2 "nodes", 1 proc each), same machine.
Set the same FOLLOWUP in both terminals: none | unshard | allreduce_avg

  # terminal A
  FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_distribute_shard_mismatch_ranks.py

  # terminal B
  FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_distribute_shard_mismatch_ranks.py

Scenario:
  Rank 0: distribute_tensor(..., [Shard(0)])
  Peer:   distribute_tensor(..., [Shard(1)])
  FOLLOWUP chooses what you try next on that state.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor


def main() -> None:
    # src_data_rank=0 takes the priority that is why we see that 0 pattern is followed. That mean we expected [0,1,2,3,4,5,6,7] arranged in [4,2] should be A getting [0,1,2,3] and B getting [1,3,5,7] but here src_rank_data is the controller .Since we choose 0 then the division will happen from there however do remember that this is not the same in residstribute also layout will be as in the node
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    followup = os.environ.get("FOLLOWUP", "none")
    dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (world_size,))

    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    if rank == 0:
        global_w.add_(100.0)
        w_sharded = distribute_tensor(global_w, mesh, [Shard(0)],src_data_rank=0)
    else:
        w_sharded = distribute_tensor(global_w, mesh, [Shard(1)],src_data_rank=0)

    print(
        f"[rank {rank}] after distribute_tensor "
        f"local={tuple(w_sharded.to_local().shape)} "
        f"global={tuple(w_sharded.shape)} placements={w_sharded.placements} "
        f"local_data={w_sharded.to_local().flatten().tolist()}"
    )

    if followup == "unshard":
        w_rep = w_sharded.redistribute(placements=[Replicate()])
        print(
            f"[rank {rank}] unshard via redistribute->Replicate "
            f"local={w_rep.to_local().flatten().tolist()}"
        )
        full = w_sharded.full_tensor()
        print(f"[rank {rank}] unshard via full_tensor={full.flatten().tolist()}")

    elif followup == "allreduce_avg":
        local = w_sharded.to_local().clone()
        print(
            f"[rank {rank}] before all_reduce AVG "
            f"shape={tuple(local.shape)} data={local.flatten().tolist()}"
        )
        dist.all_reduce(local, op=dist.ReduceOp.AVG)
        print(
            f"[rank {rank}] after all_reduce AVG "
            f"shape={tuple(local.shape)} data={local.flatten().tolist()}"
        )
        p = DTensor.from_local(w_sharded.to_local().clone(), mesh, [Partial("avg")])
        print(
            f"[rank {rank}] Partial(avg) local={p.to_local().flatten().tolist()} "
            f"placements={p.placements}"
        )
        p_rep = p.redistribute(placements=[Replicate()])
        print(
            f"[rank {rank}] Partial(avg)->Replicate "
            f"local={p_rep.to_local().flatten().tolist()}"
        )

    elif followup != "none":
        raise SystemExit(f"Unknown FOLLOWUP={followup!r}; use none|unshard|allreduce_avg")

    dist.barrier()
    if rank == 0:
        print(f"OK: distribute_tensor demo finished (FOLLOWUP={followup})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
