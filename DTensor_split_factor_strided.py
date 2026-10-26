"""
DTensor: split_factor / _StridedShard on a 2D mesh (CPU + gloo).

Needs WORLD_SIZE == 4  (mesh 2x2). Rare in 1D TP; this is the FSDP+TP-style case.

MODE:
  ltr_plain        — all ranks [Shard(0), Shard(0)]  (default left→right)
  rtl_strided      — all ranks [_StridedShard(0, split_factor=2), Shard(0)]
  mismatch_sf      — rank0 split_factor=2; others split_factor=1 (same StridedShard shape)
  mismatch_plain   — rank0 strided sf=2; others plain [Shard(0), Shard(0)]

FOLLOWUP: none | unshard

  # terminal A  (2 procs)
  MODE=rtl_strided FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=2 \
      --node_rank=0 --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_split_factor_strided.py

  # terminal B
  MODE=rtl_strided FOLLOWUP=unshard torchrun --nnodes=2 --nproc-per-node=2 \
      --node_rank=1 --master_addr=127.0.0.1 --master_port=2950 \
      DTensor_split_factor_strided.py

What to record yourself:
  - local shard data per rank under ltr_plain vs rtl_strided (who holds which rows)
  - under mismatch_*: unshard / full_tensor vs peer ranks
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, _StridedShard, distribute_tensor


def _placements_for(rank: int, mode: str):
    if mode == "ltr_plain":
        return [Shard(0), Shard(0)]
    if mode == "rtl_strided":
        return [_StridedShard(0, split_factor=2), Shard(0)]
    if mode == "mismatch_sf":
        # Same placement types; disagree on split_factor.
        sf = 2 if rank == 0 else 1
        return [_StridedShard(0, split_factor=sf), Shard(0)]
    if mode == "mismatch_plain":
        if rank == 0:
            return [_StridedShard(0, split_factor=2), Shard(0)]
        return [Shard(0), Shard(0)]
    raise SystemExit(
        f"Unknown MODE={mode!r}; use ltr_plain|rtl_strided|mismatch_sf|mismatch_plain"
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "rtl_strided")
    followup = os.environ.get("FOLLOWUP", "none")
    dist.init_process_group(backend="gloo")

    if world_size != 4:
        raise SystemExit(f"Need WORLD_SIZE=4 for mesh (2, 2); got {world_size}")

    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    # 8 rows → 4 shards of 2 rows on a 2x2 mesh when both dims shard dim 0.
    global_w = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    placements = _placements_for(rank, mode)

    print(
        f"[rank {rank}] MODE={mode} mesh={mesh.mesh.tolist()} "
        f"names={mesh.mesh_dim_names} placements={placements}"
    )
    print(
        f"[rank {rank}] global shape={tuple(global_w.shape)} "
        f"data={global_w.flatten().tolist()}"
    )

    w = distribute_tensor(global_w, mesh, placements)
    local = w.to_local()
    print(
        f"[rank {rank}] after distribute "
        f"global={tuple(w.shape)} placements={w.placements} "
        f"local_shape={tuple(local.shape)} local_data={local.flatten().tolist()}"
    )

    if followup == "unshard":
        w_rep = w.redistribute(placements=[Replicate(), Replicate()])
        print(
            f"[rank {rank}] after redistribute->Replicate*2 "
            f"local={w_rep.to_local().flatten().tolist()}"
        )
        full = w.full_tensor()
        print(f"[rank {rank}] full_tensor={full.flatten().tolist()}")
    elif followup != "none":
        raise SystemExit(f"Unknown FOLLOWUP={followup!r}; use none|unshard")

    dist.barrier()
    if rank == 0:
        print(f"OK: split_factor demo finished (MODE={mode}, FOLLOWUP={followup})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
