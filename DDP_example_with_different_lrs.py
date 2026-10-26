"""DDP + hooks: O02 — different learning rate per rank (update-rule divergence).

AllReduce still averages grads correctly; divergent weights come from each
rank applying the same .grad with a different lr. DDP does not sync parameters.

Run:
  torchrun --nproc_per_node=2 DDP_example_with_different_lrs.py
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


def register_param_hooks(module: nn.Module):
    """Autograd hooks: fire when that param's local grad is ready (per rank)."""

    def make_hook(name: str):
        def hook(grad: torch.Tensor):
            # This is the grad autograd produced on THIS rank (pre-allreduce view
            # in the hook chain). Return grad unchanged so training continues.
            log(f"param_hook {name}: local_grad_sum={grad.sum().item():.4f}")
            return grad

        return hook

    for name, param in module.named_parameters():
        param.register_hook(make_hook(name))
        log(f"registered param_hook on {name}")


def DDP_comm_hook(process_group, bucket: dist.GradBucket):
    """DDP communication hook: runs when a grad bucket is ready to sync.

    Must return Future[Tensor] (not a list). Same math as default allreduce_hook:
    divide by world_size, then allreduce(SUM) == mean.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    buf = bucket.buffer()
    log(f"comm_hook: before sync numel={buf.numel()} sum={buf.sum().item():.4f}")

    buf.div_(group.size())

    def _done(fut):
        tensor = fut.value()[0]
        log(f"comm_hook: after allreduce sum={tensor.sum().item():.4f}")
        return tensor  # Future[Tensor], not a list

    return dist.all_reduce(buf, group=group, async_op=True).get_future().then(_done)

def main():
    setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Same initial weights on every rank (important for DDP)
    torch.manual_seed(0)
    module = nn.Linear(10, 10)

    # Hooks on the bare module BEFORE wrap (sees each rank's local grad)
    register_param_hooks(module)

    model = DDP(module)  # wrap AFTER creating the module
    # Replace DDP's default allreduce with a logged version (state = process group)

    model.register_comm_hook(state=dist.group.WORLD, hook=DDP_comm_hook)
    log("DDP wrapped model + comm_hook registered")

    optimizer = optim.SGD(model.parameters(), lr=0.01 if rank == 0 else 0.02)

    # Different input per rank → different local grads; DDP allreduces them
    torch.manual_seed(rank)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    log(f"contributing batch: x_sum={x.sum().item():.4f} y_sum={y.sum().item():.4f}")

    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    log(f"local loss before sync={loss.item():.4f} (may differ per rank)")

    loss.backward()  # param_hook → then comm_hook allreduce → averaged .grad
    log(f"backward done — grads allreduced over world_size={world}")

    # .grad is the averaged value after comm_hook
    for name, param in model.module.named_parameters():
        g = param.grad.sum().item() if param.grad is not None else float("nan")
        log(f"post-backward {name}.grad_sum={g:.4f} (should match across ranks)")

    optimizer.step()

    w_sum = model.module.weight.detach().sum().item()
    b_sum = model.module.bias.detach().sum().item()
    log(f"post-step weight_sum={w_sum:.4f} bias_sum={b_sum:.4f} (will NOT match across ranks)")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception:")
        traceback.print_exc()
        raise
