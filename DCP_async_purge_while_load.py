"""keep_latest_k-style purge vs peer still loading (ops race).

MODE=healthy|purge_before_load|purge_mid_load

  DCP has no built-in ``keep_latest_k``. Train stacks often delete old step
  dirs after async save. If rank 0 purges step_0 while another rank still
  loads it → FileNotFound / CheckpointException (sometimes torn reads).

  healthy           — everyone loads step_0; no delete.
  purge_before_load — rank 0 rmtree step_0, then rank 1 loads (straggler).
  purge_mid_load    — rank 1 starts load in a thread on a larger ckpt;
                      rank 0 deletes the dir while that load runs.

Run (cuda_test):
  MODE=healthy torchrun --nproc-per-node=2 DCP_async_purge_while_load.py
  MODE=purge_before_load torchrun --nproc-per-node=2 DCP_async_purge_while_load.py
  MODE=purge_mid_load torchrun --nproc-per-node=2 DCP_async_purge_while_load.py
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def _log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def _root() -> Path:
    return Path(os.environ.get("CKPT_ROOT", "/tmp/dcp_purge_race"))


def main() -> None:
    mode = os.environ.get("MODE", "purge_before_load")
    n = int(os.environ.get("N", "2_000_000"))
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    root = _root()
    ckpt_a = root / "step_0"
    ckpt_b = root / "step_1"

    _log(f"torch={torch.__version__} MODE={mode} N={n}")

    try:
        if rank == 0:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            t = torch.arange(n, dtype=torch.float32)
            dcp.save({"w": t}, checkpoint_id=str(ckpt_a), no_dist=True)
            dcp.save({"w": t + 1}, checkpoint_id=str(ckpt_b), no_dist=True)
            _log("saved step_0 + step_1 (purge would keep only step_1)")
        dist.barrier()

        if mode == "healthy":
            out = {"w": torch.empty(n)}
            dcp.load(out, checkpoint_id=str(ckpt_a), no_dist=True)
            _log(f"loaded sum={out['w'].sum().item()} outcome: ok")
            dist.barrier()
            return

        if mode == "purge_before_load":
            # Classic straggler: purge finishes, then late load.
            if rank == 0:
                _log(f"purge: rmtree {ckpt_a}")
                shutil.rmtree(ckpt_a)
            dist.barrier()
            if rank == 1:
                out = {"w": torch.empty(n)}
                try:
                    dcp.load(out, checkpoint_id=str(ckpt_a), no_dist=True)
                    _log(f"outcome: unexpected ok sum={out['w'].sum().item()}")
                except BaseException as e:
                    _log(
                        f"outcome: loud_error {type(e).__name__}: "
                        f"{str(e).splitlines()[0][:140]}"
                    )
            dist.barrier()
            return

        if mode == "purge_mid_load":
            result: dict = {}

            def _load() -> None:
                out = {"w": torch.empty(n)}
                try:
                    dcp.load(out, checkpoint_id=str(ckpt_a), no_dist=True)
                    result["ok"] = out["w"].sum().item()
                except BaseException as e:
                    result["err"] = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"

            if rank == 1:
                th = threading.Thread(target=_load)
                th.start()
                time.sleep(0.02)  # let load open files
                dist.barrier()  # release rank 0 to purge
                th.join()
                if "ok" in result:
                    _log(f"outcome: ok (won race) sum={result['ok']}")
                else:
                    _log(f"outcome: loud_error {result.get('err')}")
            elif rank == 0:
                dist.barrier()
                _log(f"purge mid-load: rmtree {ckpt_a}")
                try:
                    shutil.rmtree(ckpt_a)
                    _log("purge: done")
                except BaseException as e:
                    _log(f"purge: {type(e).__name__}: {e}")
            else:
                dist.barrier()
            dist.barrier()
            return

        raise SystemExit(f"unknown MODE={mode}")
    finally:
        dist.barrier()
        if rank == 0 and root.exists():
            shutil.rmtree(root, ignore_errors=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
