"""DCP metadata lies + reshard load (different world size).

PHASE=save|load
MODE=healthy|lie_chunk_shape|lie_file_offset|lie_global_size

  Save with WS=2 (DTensor Shard), optionally rewrite ``.metadata``, then load
  with a *different* WS (reshard). DCP trusts metadata shapes/offsets; it does
  not re-derive them from ``.distcp`` bytes.

  lie_chunk_shape  — edit ChunkStorageMetadata sizes/offsets (logical layout)
  lie_file_offset  — edit ``storage_data`` byte offset into ``.distcp``
  lie_global_size  — edit TensorStorageMetadata.size (must match load global)

Run (cuda_test), two steps:

  CKPT=/tmp/dcp_lie_reshard MODE=healthy \\
    torchrun --nproc-per-node=2 DCP_metadata_lie_reshard.py  # PHASE defaults save+tamper

  CKPT=/tmp/dcp_lie_reshard PHASE=load \\
    torchrun --nproc-per-node=1 DCP_metadata_lie_reshard.py  # 2→1 reshard

  CKPT=/tmp/dcp_lie_reshard PHASE=load \\
    torchrun --nproc-per-node=4 DCP_metadata_lie_reshard.py  # 2→4 reshard

  Same with MODE=lie_chunk_shape|lie_file_offset|lie_global_size on the save step.
"""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata


N = 16  # global numel; divisible by 1,2,4


def _log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    print(f"[rank {rank}] {msg}", flush=True)


def _ckpt() -> Path:
    p = Path(os.environ.get("CKPT", "/tmp/dcp_lie_reshard"))
    return p


def _expected() -> torch.Tensor:
    return torch.arange(N, dtype=torch.float32)


def _save(ckpt: Path) -> None:
    rank = dist.get_rank()
    ws = dist.get_world_size()
    if rank == 0 and ckpt.exists():
        shutil.rmtree(ckpt)
    dist.barrier()

    mesh = init_device_mesh("cpu", (ws,))
    w = distribute_tensor(_expected(), mesh, [Shard(0)])
    _log(
        f"save WS={ws} local={tuple(w.to_local().shape)} "
        f"global={tuple(w.shape)} data={w.to_local().tolist()}"
    )
    dcp.save({"w": w}, checkpoint_id=str(ckpt))
    dist.barrier()
    if rank == 0:
        _log(f"saved files={sorted(p.name for p in ckpt.iterdir())}")


def _tamper(ckpt: Path, mode: str) -> None:
    if mode == "healthy":
        _log("tamper: none")
        return

    meta_path = ckpt / ".metadata"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    md = meta.state_dict_metadata["w"]
    _log(f"tamper BEFORE size={md.size} chunks={md.chunks}")
    for k, v in meta.storage_data.items():
        _log(f"  storage {k} → {v}")

    if mode == "lie_chunk_shape":
        # Keep chunk *offsets* (they key into storage_data) but shrink *sizes*
        # so the planner thinks each saved shard only covers 4 elements.
        # Real ``.distcp`` still holds 8+8; middle of the global tensor is never
        # requested → stays zero after reshard load (silent holes).
        assert len(md.chunks) >= 2, md.chunks
        md.chunks = [
            ChunkStorageMetadata(
                offsets=md.chunks[0].offsets, sizes=torch.Size([4])
            ),
            ChunkStorageMetadata(
                offsets=md.chunks[1].offsets, sizes=torch.Size([4])
            ),
        ]
        _log(
            "tamper: chunk sizes → 4+4 at same offsets "
            "(storage keys still match; coverage of [0,16) has holes)"
        )

    elif mode == "lie_file_offset":
        # Shift first chunk's file byte pointer (logical chunk meta unchanged).
        from torch.distributed.checkpoint.filesystem import _StorageInfo

        keys = sorted(
            meta.storage_data.keys(), key=lambda x: (x.fqn, x.index if x.index is not None else 0)
        )
        info = meta.storage_data[keys[0]]
        meta.storage_data[keys[0]] = _StorageInfo(
            relative_path=info.relative_path,
            offset=info.offset + 8,
            length=info.length,
            transform_descriptors=info.transform_descriptors,
        )
        _log(f"tamper: storage_data[{keys[0]!r}].offset += 8")

    elif mode == "lie_global_size":
        # Lie that global tensor is half size; load builds matching global_n.
        # Chunk offsets kept so storage_data keys still resolve; sizes halved.
        md.size = torch.Size([N // 2])
        md.chunks = [
            ChunkStorageMetadata(
                offsets=md.chunks[0].offsets, sizes=torch.Size([N // 4])
            ),
            ChunkStorageMetadata(
                offsets=md.chunks[1].offsets, sizes=torch.Size([N // 4])
            ),
        ]
        _log(f"tamper: md.size → {md.size} chunks → {md.chunks}")

    else:
        raise SystemExit(f"unknown MODE={mode}")

    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    with open(meta_path, "rb") as f:
        meta2 = pickle.load(f)
    md2 = meta2.state_dict_metadata["w"]
    _log(f"tamper AFTER size={md2.size} chunks={md2.chunks}")
    for k, v in meta2.storage_data.items():
        _log(f"  storage {k} → {v}")


def _load(ckpt: Path, mode: str) -> None:
    rank = dist.get_rank()
    ws = dist.get_world_size()
    mesh = init_device_mesh("cpu", (ws,))

    # Global size must match metadata.size (planner checks). For lie_global_size
    # we intentionally build a tensor with the *lied* size.
    global_n = N // 2 if mode == "lie_global_size" else N
    # Placeholder global then shard — values overwritten by load.
    full = torch.zeros(global_n, dtype=torch.float32)
    w = distribute_tensor(full, mesh, [Shard(0)])

    _log(
        f"load WS={ws} local={tuple(w.to_local().shape)} "
        f"global={tuple(w.shape)} (expect global_n={global_n})"
    )

    try:
        dcp.load({"w": w}, checkpoint_id=str(ckpt))
    except BaseException as e:
        msg = str(e).splitlines()[0][:180]
        _log(f"outcome: loud_error  {type(e).__name__}: {msg}")
        return

    local = w.to_local().detach().cpu()
    # Gather full for check on rank 0
    parts = [torch.zeros_like(local) for _ in range(ws)]
    dist.all_gather(parts, local)
    got = torch.cat(parts)

    if mode == "lie_global_size":
        exp = _expected()[:global_n]
    else:
        exp = _expected()

    if torch.equal(got, exp):
        outcome = "ok"
    else:
        outcome = "silent_poison"
    _log(f"loaded_full={got.tolist()}")
    _log(f"expected   ={exp.tolist()}")
    _log(f"outcome: {outcome}")


def main() -> None:
    phase = os.environ.get("PHASE", "save")
    mode = os.environ.get("MODE", "healthy")
    ckpt = _ckpt()

    dist.init_process_group(backend="gloo")
    _log(f"torch={torch.__version__} PHASE={phase} MODE={mode} CKPT={ckpt}")

    try:
        if phase == "save":
            _save(ckpt)
            dist.barrier()
            if dist.get_rank() == 0:
                _tamper(ckpt, mode)
            dist.barrier()
        elif phase == "load":
            _load(ckpt, mode)
        else:
            raise SystemExit(f"unknown PHASE={phase}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
