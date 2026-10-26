"""
Process-group backend mix probe. MODE=healthy|mix

  Mixing backends (mpi vs gloo) is unsupported. Under torchrun the MPI rank
  often hangs in MPI_Init (torchrun ≠ mpirun). Prints + heartbeat show where.

  MODE=mix TIMEOUT=8 torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=2950 PG_backend_mix_mpi_gloo.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _pick_mix_pair() -> tuple[str, str]:
    a = os.environ.get("BACKEND_A", "mpi")
    b = os.environ.get("BACKEND_B", "gloo")
    return a, b


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank} t={time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "mix")
    timeout_s = float(os.environ.get("TIMEOUT", "8"))
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "2950"))

    _log(
        rank,
        f"start available: gloo={dist.is_gloo_available()} "
        f"mpi={dist.is_mpi_available()} nccl={dist.is_nccl_available()}",
    )

    if mode == "healthy":
        backend = "gloo"
    else:
        backend_a, backend_b = _pick_mix_pair()
        backend = backend_a if rank == 0 else backend_b
        _log(rank, f"MODE=mix pair=({backend_a}, {backend_b}) → this rank backend={backend}")

    need_mpi = mode == "mix" and "mpi" in _pick_mix_pair()
    if need_mpi and not dist.is_mpi_available():
        _log(rank, "LOUD failure: is_mpi_available()=False — abort before init")
        return

    # Heartbeat so a hang shows "still inside <stage>" every 2s.
    stage = {"name": "startup"}
    stop_hb = threading.Event()

    def _heartbeat() -> None:
        while not stop_hb.wait(2.0):
            _log(rank, f"HEARTBEAT still in stage={stage['name']!r}")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    timeout = timedelta(seconds=timeout_s)
    t0 = time.time()

    try:
        # --- stage: TCPStore (rendezvous channel) ---
        stage["name"] = "TCPStore_create"
        _log(
            rank,
            f"stage=TCPStore_create master={master}:{port} "
            f"is_master={rank == 0} timeout={timeout_s}s",
        )
        store = dist.TCPStore(
            master,
            port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=timeout,
        )
        _log(rank, f"stage=TCPStore_ok elapsed={time.time() - t0:.2f}s")

        # --- stage: explicit KV ping so we know both reached the store ---
        stage["name"] = "store_kv_ping"
        key = f"ping/rank{rank}"
        _log(rank, f"stage=store_kv_ping set {key}")
        store.set(key, b"1")
        peer = 1 - rank if world_size == 2 else 0
        peer_key = f"ping/rank{peer}"
        _log(rank, f"stage=store_kv_ping waiting get {peer_key}")
        store.get(peer_key)
        _log(rank, f"stage=store_kv_ping_ok peer seen elapsed={time.time() - t0:.2f}s")

        if backend == "mpi":
            _log(
                rank,
                "WARNING: next stage enters MPI_Init via ProcessGroupMPI "
                "(often hangs under torchrun — not an MPI launcher)",
            )

        # --- stage: init_process_group (store already up; backend constructed here) ---
        stage["name"] = f"init_process_group(backend={backend})"
        _log(rank, f"stage=init_process_group backend={backend} …")
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )
        _log(rank, f"stage=init_ok elapsed={time.time() - t0:.2f}s")

        stage["name"] = "all_reduce"
        _log(rank, "stage=all_reduce …")
        x = torch.ones(2)
        dist.all_reduce(x)
        _log(rank, f"stage=all_reduce_ok sum={x.sum().item()}")

        stage["name"] = "barrier"
        _log(rank, "stage=barrier …")
        dist.barrier()
        _log(rank, "stage=barrier_ok")

        if rank == 0:
            _log(rank, f"OK: backend demo finished (MODE={mode})")
        dist.destroy_process_group()
    except Exception as e:
        _log(
            rank,
            f"LOUD failure in stage={stage['name']!r} after {time.time() - t0:.2f}s: "
            f"{type(e).__name__}: {e}",
        )
        sys.exit(1)
    finally:
        stop_hb.set()


if __name__ == "__main__":
    main()
