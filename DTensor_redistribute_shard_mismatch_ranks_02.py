"""
DTensor: redistribute the same logical tensor to different placements per rank
(CPU + gloo), then optionally unshard or all-reduce-average.

Two terminals (2 "nodes", 1 proc each), same machine.
Set the same FOLLOWUP in both terminals: none | unshard | allreduce_avg

  # terminal A
  FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_redistribute_shard_mismatch_ranks.py

  # terminal B
  FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=29500 \
      DTensor_redistribute_shard_mismatch_ranks.py

Scenario:
  Start agreed on Replicate; then
  rank 0: redistribute([Shard(0)]), peer: redistribute([Shard(1)]).
  FOLLOWUP chooses what you try next on that state.

What to record yourself (no assumptions):
  - shapes / placements / values after the mismatched redistribute
  - for unshard: redistribute([Replicate()]) and full_tensor()
  - for allreduce_avg: local all_reduce(AVG) and Partial("avg") -> Replicate
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor import DTensor


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    followup = os.environ.get("FOLLOWUP", "none")
    dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (world_size,))

    global_w = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    w = distribute_tensor(global_w, mesh, [Replicate()])
    print(
        f"[rank {rank}] start Replicate local={tuple(w.to_local().shape)} "
        f"global={tuple(w.shape)} placements={w.placements}"
    )

    target = [Shard(0)] if rank == 0 else [Shard(1)]
    print(f"[rank {rank}] redistribute target={target}")
    w_sharded = w.redistribute(placements=target)
    print(
        f"[rank {rank}] after redistribute "
        f"local={tuple(w_sharded.to_local().shape)} "
        f"global={tuple(w_sharded.shape)} placements={w_sharded.placements} "
        f"local_data={w_sharded.to_local().flatten().tolist()}"
    )

    if followup == "unshard":
        # Path A: ask DTensor to gather back to a full replica.
        w_rep = w_sharded.redistribute(placements=[Replicate()])
        print(
            f"[rank {rank}] unshard via redistribute->Replicate "
            f"local={w_rep.to_local().flatten().tolist()} placements={w_rep.placements}"
        )
        # Path B: materialize the full logical tensor explicitly.
        full = w_sharded.full_tensor()
        print(f"[rank {rank}] unshard via full_tensor={full.flatten().tolist()}")

    elif followup == "allreduce_avg":
        # Path A: classic process-group all-reduce average on the local shard.
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

        # Path B: express "this local is a partial / avg contribution" in DTensor.
        # Uses the same local buffer shape each rank currently holds.
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
        print(f"OK: redistribute demo finished (FOLLOWUP={followup})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
