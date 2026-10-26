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

def malformed_comm_hook(process_group, bucket: dist.GradBucket):
    """Intentionally mess with buf shape/dtype, then allreduce.

    DDP copies the returned Future[Tensor] back into the bucket grads, so the
    final tensor MUST have the same numel (and usually same dtype) as
    bucket.buffer(). You can cat / cast for the collective, but slice/cast back
    before returning — otherwise the reducer crashes.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    buf = bucket.buffer()
    orig_numel = buf.numel()
    orig_dtype = buf.dtype

    log(
        f"comm_hook: original buf shape={tuple(buf.shape)} "
        f"dtype={buf.dtype} numel={orig_numel} sum={buf.sum().item():.4f}"
    )

    # --- pick one (or both) ways to "malform" before the collective -----------
    # 1) change dtype for the wire (fp32 -> fp16), cast back after
    work = buf.to(torch.bfloat16)

    # # 2) change shape via cat (pad a few junk elements), slice back after
    # pad = torch.zeros(4, device=work.device, dtype=work.dtype)
    # work = torch.cat([work.reshape(-1), pad], dim=0)  # 1-D, longer than orig
    # log(f"comm_hook: malformed work shape={tuple(work.shape)} dtype={work.dtype}")

    # mean allreduce on the malformed tensor (every rank must cat/cast the same way)
    work.div_(group.size())

    def _done(fut):
        tensor = fut.value()[0]
        # undo malformation so DDP can scatter grads back into params
        tensor = tensor[:orig_numel].to(orig_dtype)
        log(
            f"comm_hook: restored shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"sum={tensor.sum().item():.4f}"
        )
        return tensor  # Future[Tensor], same numel as original buf

    return dist.all_reduce(work, group=group, async_op=True).get_future().then(_done)


def main():
    setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Same initial weights on every rank (important for DDP)
    torch.manual_seed(0)
    model = nn.Linear(10, 10)
    register_param_hooks(model)
    model = DDP(model)  # wrap AFTER creating the module
    
    if rank == 0:
        model.register_comm_hook(state=dist.group.WORLD, hook=malformed_comm_hook)
        log("DDP wrapped model + malformed comm_hook registered")

    # model.register_comm_hook(state=dist.group.WORLD, hook=malformed_comm_hook)
    log("DDP wrapped model + malformed comm_hook registered")

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Different input per rank → different local grads; DDP allreduces them
    torch.manual_seed(rank)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    log(f"contributing batch: x_sum={x.sum().item():.4f} y_sum={y.sum().item():.4f}")

    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    log(f"local loss before sync={loss.item():.4f} (may differ per rank)")

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
