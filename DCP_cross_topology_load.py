"""DCP cross-topology load probe (no mesh validation on reshard).

PHASE=save|load
MODE=baseline|mesh_factor|mesh_axis_swap|shard_dim

  DCP ``.metadata`` stores **global tensor chunks** (offsets/sizes), not
  TP×CP×DP mesh dim names. Load builds read plans by intersecting *your*
  current local shards with those chunks. There is **no** check that the
  load mesh / placement matches the save mesh ("unvalidated reshard").

  baseline        — save WS=N, load WS=M; expect full-tensor equality.
  mesh_factor    — WS=4; save (tp=4,) Shard(0); load (tp=2,dp=2) Shard+Replicate.
  mesh_axis_swap  — WS=4; same mesh names (tp,dp), but change placements in
                    ``distribute_tensor``: save Shard on tp, load Shard on dp.
                    (Renaming axes alone is a no-op on a (2,2) grid — we don't
                    do that.) Locals move across ranks; full_tensor may still match.
  shard_dim       — WS=2; save Shard(0) on rows; load Shard(1) on cols.

  Pass/fail here is **value** equality only — that is exactly the gap:
  DCP will not refuse a cross-topology load if chunk boxes still cover.

Run (cuda_test / gloo CPU):

  # baseline: N=2 → M=1
  CKPT=/tmp/dcp_xtopo MODE=baseline \\
    torchrun --nproc-per-node=2 DCP_cross_topology_load.py
  CKPT=/tmp/dcp_xtopo MODE=baseline PHASE=load \\
    torchrun --nproc-per-node=1 DCP_cross_topology_load.py

  # baseline: N=2 → M=4
  CKPT=/tmp/dcp_xtopo MODE=baseline PHASE=load \\
    torchrun --nproc-per-node=4 DCP_cross_topology_load.py

  # mesh_factor / mesh_axis_swap need WS=4 both phases
  CKPT=/tmp/dcp_xtopo MODE=mesh_factor \\
    torchrun --nproc-per-node=4 DCP_cross_topology_load.py
  CKPT=/tmp/dcp_xtopo MODE=mesh_factor PHASE=load \\
    torchrun --nproc-per-node=4 DCP_cross_topology_load.py

  Same with MODE=mesh_axis_swap.

  CKPT=/tmp/dcp_xtopo MODE=shard_dim \\
    torchrun --nproc-per-node=2 DCP_cross_topology_load.py
  CKPT=/tmp/dcp_xtopo MODE=shard_dim PHASE=load \\
    torchrun --nproc-per-node=2 DCP_cross_topology_load.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor


ROWS, COLS = 8, 4  # global weight shape; 8 divisible by 1,2,4


def _log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def _ckpt() -> Path:
    return Path(os.environ.get("CKPT", "/tmp/dcp_xtopo"))


def _golden() -> torch.Tensor:
    return torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)


def _build_mesh(mode: str, phase: str):
    """Return (mesh, placements) for this rank's world."""
    ws = dist.get_world_size()
    if mode == "baseline":
        # 1D DP/TP-agnostic shard on row dim.
        mesh = init_device_mesh("cpu", (ws,), mesh_dim_names=("x",))
        return mesh, [Shard(0)]

    if mode == "mesh_factor":
        if ws != 4:
            raise SystemExit("mesh_factor requires WORLD_SIZE=4")
        if phase == "save":
            mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("tp",))
            return mesh, [Shard(0)]
        # load: factor 4 = 2×2
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "dp"))
        # Shard on TP only; replicate on DP (common FSDP+TP style).
        return mesh, [Shard(0), Replicate()]

    if mode == "mesh_axis_swap":
        if ws != 4:
            raise SystemExit(
                "mesh_axis_swap needs WORLD_SIZE=4 (2x2 mesh), e.g.\n"
                "  torchrun --nproc-per-node=4 DCP_cross_topology_load.py\n"
                "  MODE=mesh_axis_swap PHASE=load torchrun --nproc-per-node=4 ..."
            )
        # Same mesh layout and names; only placements passed to distribute_tensor
        # change — that is the real topology story change.
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "dp"))
        if phase == "save":
            return mesh, [Shard(0), Replicate()]  # shard rows along tp
        return mesh, [Replicate(), Shard(0)]  # shard rows along dp instead

    if mode == "shard_dim":
        if ws != 2:
            raise SystemExit("shard_dim requires WORLD_SIZE=2")
        mesh = init_device_mesh("cpu", (ws,), mesh_dim_names=("tp",))
        if phase == "save":
            return mesh, [Shard(0)]  # row-parallel style
        return mesh, [Shard(1)]  # column-parallel style — different TP story

    raise SystemExit(f"unknown MODE={mode}")


def _save(ckpt: Path, mode: str) -> None:
    rank = dist.get_rank()
    if rank == 0 and ckpt.exists():
        shutil.rmtree(ckpt)
    dist.barrier()

    mesh, placements = _build_mesh(mode, "save")
    w = distribute_tensor(_golden(), mesh, placements)
    _log(
        f"SAVE mesh={tuple(mesh.shape)} names={mesh.mesh_dim_names} "
        f"placements={placements} local={tuple(w.to_local().shape)} "
        f"coord={mesh.get_coordinate()}"
    )
    dcp.save({"w": w}, checkpoint_id=str(ckpt))
    dist.barrier()
    if rank == 0:
        _log(f"saved {sorted(p.name for p in ckpt.iterdir())}")


def _load(ckpt: Path, mode: str) -> None:
    mesh, placements = _build_mesh(mode, "load")
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)
    w = distribute_tensor(empty, mesh, placements)
    _log(
        f"LOAD mesh={tuple(mesh.shape)} names={mesh.mesh_dim_names} "
        f"placements={placements} local={tuple(w.to_local().shape)} "
        f"coord={mesh.get_coordinate()}"
    )

    try:
        dcp.load({"w": w}, checkpoint_id=str(ckpt))
    except BaseException as e:
        _log(f"outcome: loud_error {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return

    local = w.to_local().detach()
    full = w.full_tensor().detach()
    exp = _golden()
    # Note: no DCP API validates save-mesh vs load-mesh; we only check values.
    ok = torch.equal(full, exp)
    _log(f"local={local.flatten().tolist()[:16]}{'...' if local.numel() > 16 else ''}")
    _log(f"full_tensor equal_golden={ok}")
    if not ok:
        _log(f"full={full.flatten().tolist()}")
        _log(f"exp ={exp.flatten().tolist()}")
        _log("outcome: silent_poison (or mismatched reconstruction)")
    else:
        _log(
            "outcome: ok (value equality). "
            "DCP did NOT validate TP×CP×DP mesh; it only matched chunk boxes."
        )


def main() -> None:
    phase = os.environ.get("PHASE", "save")
    mode = os.environ.get("MODE", "baseline")
    ckpt = _ckpt()

    dist.init_process_group(backend="gloo")
    _log(f"torch={torch.__version__} PHASE={phase} MODE={mode} WS={dist.get_world_size()}")

    try:
        if phase == "save":
            _save(ckpt, mode)
        elif phase == "load":
            _load(ckpt, mode)
        else:
            raise SystemExit(f"unknown PHASE={phase}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
