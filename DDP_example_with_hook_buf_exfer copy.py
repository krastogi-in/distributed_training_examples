"""R03 + Join: who is authoritative for DDP buffers?

Without Join: buffers always broadcast from rank 0.
With Join:    DDP picks max(active ranks) so rank 0 can be skipped if it already
              joined with stale buffers (see distributed.py _sync_buffers).

Demo: rank 0 poisons running_mean=9999; under Join the next forward syncs FROM
rank 1 (non-zero leader), so rank 0's poison is overwritten — not installed.

Uneven inputs: rank 0 runs fewer steps and joins early; Join shadows collectives
so rank 1 can finish. Set THROW=1 to *detect* uneven inputs via exception.

Run:
  torchrun --nproc_per_node=2 DDP_example_with_hook_buf_exfer.py
  THROW=1 torchrun --nproc_per_node=2 DDP_example_with_hook_buf_exfer.py
"""
import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP


def env_rank_info():
    return {
        "host": socket.gethostname(),
        "rank": int(os.environ.get("RANK", -1)),
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "world_size": int(os.environ.get("WORLD_SIZE", -1)),
        "master": f"{os.environ.get('MASTER_ADDR', '?')}:{os.environ.get('MASTER_PORT', '?')}",
    }


def log(msg: str):
    info = env_rank_info()
    print(
        f"[host={info['host']} rank={info['rank']} "
        f"local_rank={info['local_rank']}/{info['world_size']}] {msg}",
        flush=True,
    )


def setup():
    log(f"joining process group via {env_rank_info()['master']} (backend=gloo)")
    dist.init_process_group(backend="gloo")


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    setup()
    rank = dist.get_rank()
    throw = os.environ.get("THROW", "0") == "1"

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    ddp = DDP(model, forward_sync_buffers=True)
    bn = model[1]

    # Warmup (both ranks still active)
    ddp(torch.randn(4, 8))
    log(f"warmup running_mean[:3]={bn.running_mean[:3].tolist()}")

    # Rank 0 installs attacker/stale buffers; rank 1 leaves its post-warmup values
    if rank == 0:
        with torch.no_grad():
            bn.running_mean.fill_(9999.0)
        log("rank 0 poisoned running_mean=9999 (stale/attacker-chosen)")
    else:
        log(f"rank {rank} local running_mean[:3]={bn.running_mean[:3].tolist()}")

    dist.barrier()

    # Uneven inputs: rank 0 joins after 1 more step; rank 1 does 3.
    # Join shadows collectives for the early leaver.
    steps = 1 if rank == 0 else 3
    log(f"Join(enable=True, throw_on_early_termination={throw}); local steps={steps}")

    with Join([ddp], enable=True, throw_on_early_termination=throw):
        for i in range(steps):
            out = ddp(torch.randn(4, 8))
            out.sum().backward()
            # Under Join, buffer sync uses max(active rank), not always 0.
            # After rank 0's first (only) step it joins; further syncs are led
            # by still-active ranks (here rank 1).
            log(f"step {i} running_mean[:3]={bn.running_mean[:3].tolist()}")

    # Final Join post_hook broadcasts model from last joiner (tie → larger rank)
    log(f"after Join running_mean[:3]={bn.running_mean[:3].tolist()}")
    mean_abs = bn.running_mean.abs().mean().item()
    if mean_abs > 1000.0:
        log("NOTE: huge mean → rank-0 poison still present (would mean rank 0 led buffers)")
    else:
        log("R03/Join: buffers reflect non-zero (active/last) leader, not rank-0 poison")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception:")
        traceback.print_exc()
        raise
