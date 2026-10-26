"""Minimal DistributedDataParallel example (CPU / gloo).

Run:
  torchrun --nproc_per_node=2 DDP_example_1.py
"""
import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def env_rank_info():
    """torchrun sets these before your script starts."""
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
    info = env_rank_info()
    log(f"joining process group via {info['master']} (backend=gloo)")
    dist.init_process_group(backend="gloo")
    log("process group ready — this rank is in the job")


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        log("destroyed process group")


def main():
    setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Same initial weights on every rank (important for DDP)
    torch.manual_seed(0)
    model = nn.Linear(10, 10)
    model = DDP(model)  # wrap AFTER creating the module
    log("DDP wrapped model")

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Different input per rank → different local grads; DDP allreduces them
    torch.manual_seed(rank)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    log(f"contributing batch: x_sum={x.sum().item():.4f} y_sum={y.sum().item():.4f}")

    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    log(f"local loss before sync={loss.item():.4f} (may differ per rank)")

    # # if rank == 0:
    # #     loss = loss * 0.0
    # #     print(loss)

    # if rank == 0:
    #     loss = loss * float('nan')
    #     print(loss)
    # print(loss)

    loss.backward()  # gradients allreduced across all ranks here
    log(f"backward done — grads allreduced over world_size={world}")

    optimizer.step()
    log("optimizer.step done (local weight update)")

    # After step, every rank should hold the same weights
    w_sum = model.module.weight.detach().sum().item()
    b_sum = model.module.bias.detach().sum().item()
    log(f"post-step weight_sum={w_sum:.4f} bias_sum={b_sum:.4f} (should match across ranks)")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Elastic hides tracebacks unless each rank prints its own
        log("FAILED with exception:")
        traceback.print_exc()
        raise
