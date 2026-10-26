"""Async DCP save + mutate params while persist is in flight (quick try).

MODE=default|nocopy

  default — stock ``dcp.async_save`` (stages a CPU copy first). Mutate the
            live tensor right after the call returns, then wait for upload.
            Expect: checkpoint still has the *pre-mutate* values (safe).
  nocopy  — custom stager that returns the same dict (no copy). Mutate while
            the background writer runs. Expect: silent poison / torn values.

Run (cuda_test, single process):
  conda run -n cuda_test python -W ignore DCP_async_mutate_inflight.py
  MODE=nocopy conda run -n cuda_test python -W ignore DCP_async_mutate_inflight.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemWriter


class NoCopyStager:
    """Intentionally unsafe: stage() aliases the training state_dict."""

    _synchronize_after_execute = True

    @property
    def should_synchronize_after_execute(self) -> bool:
        return True

    def stage(self, state_dict):
        return state_dict

    def synchronize_staging(self) -> None:
        pass

    def close(self) -> None:
        pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    mode = os.environ.get("MODE", "default")
    n = int(os.environ.get("N", "8_000_000"))  # large so upload outlasts mutate loop
    ckpt = Path(tempfile.mkdtemp(prefix="dcp_async_"))
    t = torch.arange(n, dtype=torch.float32)
    before = t.clone()
    state = {"w": t}
    poison = 999.0

    _log(f"torch={torch.__version__} MODE={mode} N={n}")
    _log(f"before sum={before.sum().item()}")

    writer = FileSystemWriter(str(ckpt), overwrite=True)
    try:
        if mode == "default":
            fut = dcp.async_save(
                state, checkpoint_id=str(ckpt), storage_writer=writer, no_dist=True
            )
        elif mode == "nocopy":
            fut = dcp.async_save(
                state,
                checkpoint_id=str(ckpt),
                storage_writer=writer,
                async_stager=NoCopyStager(),
                no_dist=True,
            )
        else:
            raise SystemExit(f"unknown MODE={mode}")

        # Hammer the live tensor until the background save finishes.
        flips = 0
        while not fut.done():
            t.fill_(poison)
            flips += 1
            time.sleep(0)
        fut.result()
        _log(f"mutated while in-flight flips={flips} live_sum={t.sum().item()}")

        loaded = {"w": torch.empty_like(before)}
        dcp.load(loaded, checkpoint_id=str(ckpt), no_dist=True)
        got = loaded["w"]
        _log(f"loaded sum={got.sum().item()}  before_sum={before.sum().item()}")

        if torch.equal(got, before):
            _log("outcome: ok (matches pre-mutate — staging copy protected you)")
        elif torch.allclose(got, torch.full_like(before, poison)):
            _log("outcome: silent_poison (all post-mutate values)")
        else:
            # Mixed / partial — classic torn write.
            n_poison = int((got == poison).sum().item())
            n_ok = int((got == before).sum().item())
            _log(
                f"outcome: torn/corrupt  n_pre={n_ok} n_poison={n_poison} "
                f"other={got.numel() - n_ok - n_poison}"
            )
    finally:
        shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    main()
