"""
Per-rank Gloo env / timeout divergence. MODE=healthy|timeout|ifname

  Trust model: ranks must agree on process-group setup. Different timeouts or
  GLOO_SOCKET_IFNAME is ops skew (not a remote vuln) — one rank errors/hangs first.

  healthy:  both timeout=30s, GLOO_SOCKET_IFNAME=lo (or unset)
  timeout:  rank0 init timeout=SHORT (default 5s), rank1=LONG (default 60s);
            rank1 sleeps DELAY (default 15s) before allreduce → rank0 should
            time out first while rank1 is still delayed / waiting.
  ifname:   rank0 GLOO_SOCKET_IFNAME=lo, rank1=BAD_IFACE (default eth999);
            set before init — expect bind/connect failure or hang on init.

  Env knobs: SHORT=5 LONG=60 DELAY=15 BAD_IFACE=eth999

  Terminal 0:
    MODE=timeout torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 GLOO_env_timeout_diverge.py
  Terminal 1:
    MODE=timeout torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 GLOO_env_timeout_diverge.py
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    mode = os.environ.get("MODE", "timeout")
    short_s = float(os.environ.get("SHORT", "5"))
    long_s = float(os.environ.get("LONG", "60"))
    delay_s = float(os.environ.get("DELAY", "15"))
    bad_iface = os.environ.get("BAD_IFACE", "eth999")

    # ifname must be set before init_process_group creates the Gloo device.
    if mode == "ifname":
        iface = "lo" if rank == 0 else bad_iface
        os.environ["GLOO_SOCKET_IFNAME"] = iface
        print(f"[rank {rank}] MODE={mode} set GLOO_SOCKET_IFNAME={iface}")
    elif mode == "healthy":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        print(
            f"[rank {rank}] MODE={mode} GLOO_SOCKET_IFNAME="
            f"{os.environ.get('GLOO_SOCKET_IFNAME')}"
        )

    if mode == "timeout":
        timeout = timedelta(seconds=short_s if rank == 0 else long_s)
    else:
        timeout = timedelta(seconds=30)

    print(f"[rank {rank}] MODE={mode} init gloo timeout={timeout.total_seconds():.1f}s")
    t0 = time.time()
    try:
        dist.init_process_group(backend="gloo", timeout=timeout)
    except Exception as e:
        print(
            f"[rank {rank}] INIT FAILED after {time.time() - t0:.2f}s: "
            f"{type(e).__name__}: {e}"
        )
        raise

    print(f"[rank {rank}] init_ok after {time.time() - t0:.2f}s")

    x = torch.ones(2) * (rank + 1)

    if mode == "timeout" and rank == 1:
        print(f"[rank {rank}] sleeping {delay_s}s before allreduce "
              f"(peer short timeout={short_s}s)")
        time.sleep(delay_s)

    print(f"[rank {rank}] entering allreduce at t+{time.time() - t0:.2f}s")
    try:
        dist.all_reduce(x)
        print(
            f"[rank {rank}] allreduce_ok x={x.tolist()} "
            f"at t+{time.time() - t0:.2f}s"
        )
    except Exception as e:
        print(
            f"[rank {rank}] ALLREDUCE FAILED after {time.time() - t0:.2f}s: "
            f"{type(e).__name__}: {e}"
        )
        raise

    dist.barrier()
    if rank == 0:
        print(f"OK: Gloo env/timeout demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
