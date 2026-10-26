"""DCP pickle / torch.load surface audit (harmless marker canary).

Trust model (same as torch.distributed): DCP assumes a trusted job + trusted
store. This script only maps *where* deserialization happens and whether a
``__reduce__`` canary that touches a local marker file runs.

Surfaces (torch 2.8 / DCP filesystem planner):

  S1  FileSystemReader.read_metadata     → pickle.load(<dir>/.metadata)
  S2  DefaultLoadPlanner.load_bytes      → torch.load(..., weights_only=False)
                                          (BYTE_IO / non-tensor state_dict values)
  S3  FileSystemReader.read_data tensor  → torch.load(..., weights_only=True)
  S4  format_utils.torch_save_to_dcp     → torch.load(..., weights_only=False)
  S5  format_utils.BroadcastingTorchSaveReader
                                         → torch.load(..., weights_only=False)
  S6  _pg_transport.recv_checkpoint      → pickle.loads(meta bytes)
                                          (newer trees; same sink exercised here)

Run (single process, no torchrun needed):
  python DCP_pickle_surface_audit.py
  SURFACES=S1,S2 python DCP_pickle_surface_audit.py
"""

from __future__ import annotations

import io
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    DynamicMetaLoadPlanner,
    torch_save_to_dcp,
)


MARKER_ROOT = Path(
    os.environ.get("DCP_CANARY_DIR", tempfile.mkdtemp(prefix="dcp_canary_"))
)


def _marker(name: str) -> Path:
    return MARKER_ROOT / f"{name}.touched"


def _touch(path: str) -> None:
    Path(path).write_text("canary\n", encoding="utf-8")


class Canary:
    """Harmless pickle gadget: on unpickle, create ``marker`` only."""

    def __init__(self, marker: str):
        self.marker = marker

    def __reduce__(self):
        return (_touch, (self.marker,))


def _clear(marker: Path) -> None:
    if marker.exists():
        marker.unlink()


def _result(surface: str, marker: Path, err: BaseException | None = None) -> None:
    fired = marker.exists()
    status = "FIRED" if fired else "blocked/no-exec"
    if err is None:
        extra = ""
    else:
        msg = str(err).split("\n", 1)[0][:120]
        extra = f"  err={type(err).__name__}: {msg}"
    print(f"[{surface}] {status}  marker={marker}{extra}", flush=True)


def surface_s1_metadata_pickle_load() -> None:
    """S1: replace .metadata with pickled Canary; FileSystemReader.read_metadata."""
    marker = _marker("S1")
    _clear(marker)
    d = Path(tempfile.mkdtemp(prefix="dcp_s1_"))
    err: BaseException | None = None
    try:
        with open(d / ".metadata", "wb") as f:
            pickle.dump(Canary(str(marker)), f)
        FileSystemReader(str(d)).read_metadata()
    except BaseException as e:
        err = e
    finally:
        shutil.rmtree(d, ignore_errors=True)
    _result("S1 .metadata pickle.load", marker, err)


def surface_s2_bytes_torch_load() -> None:
    """S2: non-tensor BYTE_IO value → DefaultLoadPlanner.load_bytes torch.load."""
    marker = _marker("S2")
    _clear(marker)
    d = Path(tempfile.mkdtemp(prefix="dcp_s2_"))
    err: BaseException | None = None
    try:
        state = {
            "w": torch.ones(2),
            "extra": Canary(str(marker)),  # becomes BYTE_IO via torch.save
        }
        dcp.save(state, checkpoint_id=str(d), no_dist=True)
        # Load into a dict that expects the same keys (placeholder for bytes).
        load_sd = {"w": torch.zeros(2), "extra": None}
        dcp.load(load_sd, checkpoint_id=str(d), no_dist=True)
    except BaseException as e:
        err = e
    finally:
        shutil.rmtree(d, ignore_errors=True)
    _result("S2 BYTE_IO torch.load(weights_only=False)", marker, err)


def surface_s3_tensor_weights_only() -> None:
    """S3: same kwargs as FileSystemReader.read_data tensor branch.

    ``torch.load(..., map_location="cpu", weights_only=True)`` — canary must
    not run. (Shard files are offset slices inside a multi-item stream, so we
    call the sink directly rather than rewriting whole files.)
    """
    marker = _marker("S3")
    _clear(marker)
    err: BaseException | None = None
    try:
        buf = io.BytesIO()
        torch.save(Canary(str(marker)), buf)
        buf.seek(0)
        torch.load(buf, map_location="cpu", weights_only=True)
    except BaseException as e:
        err = e
    _result("S3 tensor torch.load(weights_only=True)", marker, err)


def surface_s4_torch_save_to_dcp() -> None:
    """S4: format_utils.torch_save_to_dcp → torch.load(weights_only=False)."""
    marker = _marker("S4")
    _clear(marker)
    src = Path(tempfile.mkdtemp(prefix="dcp_s4_src_")) / "model.pt"
    dst = Path(tempfile.mkdtemp(prefix="dcp_s4_dst_"))
    err: BaseException | None = None
    try:
        torch.save({"extra": Canary(str(marker)), "w": torch.ones(2)}, src)
        torch_save_to_dcp(str(src), str(dst))
    except BaseException as e:
        err = e
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)
    _result("S4 torch_save_to_dcp torch.load", marker, err)


def surface_s5_broadcasting_torch_save_reader() -> None:
    """S5: BroadcastingTorchSaveReader.read_data coordinator torch.load.

    Needs a process group. 1-rank gloo + DynamicMetaLoadPlanner (docs pairing).
    Canary lives under an unused key; torch.load still unpickles the whole file.
    """
    marker = _marker("S5")
    _clear(marker)
    src = Path(tempfile.mkdtemp(prefix="dcp_s5_")) / "model.pt"
    err: BaseException | None = None
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29771")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    started = False
    try:
        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=0, world_size=1)
            started = True
        torch.save({"w": torch.ones(2), "extra": Canary(str(marker))}, src)
        state = {"w": torch.zeros(2)}
        dcp.load(
            state,
            storage_reader=BroadcastingTorchSaveReader(),
            planner=DynamicMetaLoadPlanner(),
            checkpoint_id=str(src),
        )
    except BaseException as e:
        err = e
    finally:
        if started and dist.is_initialized():
            dist.destroy_process_group()
        shutil.rmtree(src.parent, ignore_errors=True)
    _result("S5 BroadcastingTorchSaveReader torch.load", marker, err)


def surface_s6_pg_transport_pickle_loads() -> None:
    """S6: same sink as _pg_transport.recv_checkpoint meta: pickle.loads(bytes)."""
    marker = _marker("S6")
    _clear(marker)
    err: BaseException | None = None
    try:
        buf = pickle.dumps(Canary(str(marker)))
        # Mirror: meta = pickle.loads(buf.cpu().numpy().tobytes())
        pickle.loads(buf)
    except BaseException as e:
        err = e
    _result("S6 _pg_transport-style pickle.loads", marker, err)


SURFACES = {
    "S1": surface_s1_metadata_pickle_load,
    "S2": surface_s2_bytes_torch_load,
    "S3": surface_s3_tensor_weights_only,
    "S4": surface_s4_torch_save_to_dcp,
    "S5": surface_s5_broadcasting_torch_save_reader,
    "S6": surface_s6_pg_transport_pickle_loads,
}


def main() -> None:
    want = os.environ.get("SURFACES", "S1,S2,S3,S4,S5,S6")
    keys = [k.strip().upper() for k in want.split(",") if k.strip()]
    print(
        f"torch={torch.__version__}  markers under {MARKER_ROOT}\n"
        f"running: {keys}",
        flush=True,
    )
    for k in keys:
        if k not in SURFACES:
            print(f"[?] unknown surface {k}", flush=True)
            continue
        print(f"--- {k} ---", flush=True)
        SURFACES[k]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
