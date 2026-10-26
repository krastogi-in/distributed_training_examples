"""DDP: O04 — per-rank RNG / dropout policy skew (local stochasticity).

DDP does not synchronize dropout masks or RNG state. Each rank draws its own
masks during forward. Local losses and local grads differ; backward still
AllReduces to one mean .grad. If every rank applies the same optimizer.step(),
weights stay matched — skew mostly adds noise / hurts reproducibility.

Combined with O01 (one rank skips optimizer.step), the same skew plus
divergent weights makes the next forward even more inconsistent.

Run:
  torchrun --nproc_per_node=2 DDP_example_with_rng_dropout_skew.py
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


class DropoutLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.fc1(x)))


def register_param_hooks(module: nn.Module):
    def make_hook(name: str):
        def hook(grad: torch.Tensor):
            log(f"param_hook {name}: local_grad_sum={grad.sum().item():.4f}")
            return grad

        return hook

    for name, param in module.named_parameters():
        param.register_hook(make_hook(name))


def DDP_comm_hook(process_group, bucket: dist.GradBucket):
    group = process_group if process_group is not None else dist.group.WORLD
    buf = bucket.buffer()
    log(f"comm_hook: before sync numel={buf.numel()} sum={buf.sum().item():.4f}")
    buf.div_(group.size())

    def _done(fut):
        tensor = fut.value()[0]
        log(f"comm_hook: after allreduce sum={tensor.sum().item():.4f}")
        return tensor

    return dist.all_reduce(buf, group=group, async_op=True).get_future().then(_done)


def skew_dropout_rng(rank: int):
    """Intentional per-rank RNG — not broadcast; simulates policy / repro skew."""
    torch.manual_seed(10_000 + rank)
    log(f"O04: manual_seed({10_000 + rank}) before forward (dropout masks differ per rank)")


def run_forward_backward(model, optimizer, x, y, world: int):
    optimizer.zero_grad()
    skew_dropout_rng(dist.get_rank())
    model.train()
    loss = nn.functional.mse_loss(model(x), y)
    log(f"local loss before sync={loss.item():.4f} (differs with dropout skew)")
    loss.backward()
    log(f"backward done — grads allreduced over world_size={world}")
    for name, param in model.module.named_parameters():
        g = param.grad.sum().item() if param.grad is not None else float("nan")
        log(f"post-backward {name}.grad_sum={g:.4f} (should match across ranks)")


def log_weight_sums(model: DDP, label: str):
    params = dict(model.module.named_parameters())
    weight = params["fc2.weight"]
    bias = params["fc2.bias"]
    log(
        f"{label} fc2 weight_sum={weight.detach().sum().item():.4f} "
        f"bias_sum={bias.detach().sum().item():.4f}"
    )


def main():
    setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    torch.manual_seed(0)
    module = DropoutLinear()
    register_param_hooks(module)
    model = DDP(module)
    model.register_comm_hook(state=dist.group.WORLD, hook=DDP_comm_hook)
    log("DDP wrapped model + comm_hook registered")

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Identical batch on every rank — only dropout RNG differs (O04).
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    log(f"shared batch: x_sum={x.sum().item():.4f} y_sum={y.sum().item():.4f}")

    log("--- part 1: O04 only — both ranks step; weights should still match ---")
    run_forward_backward(model, optimizer, x, y, world)
    optimizer.step()
    log_weight_sums(model, "post-step (symmetric):")

    dist.barrier()
    if rank == 0:
        log("(rank 0) fc2 sums should equal rank 1 — same synced .grad, same W₀")

    log("--- part 2: O04 + O01 — rank 0 skips step; weights diverge ---")
    run_forward_backward(model, optimizer, x, y, world)
    if rank == 0:
        log("O01: rank 0 skips optimizer.step()")
    else:
        optimizer.step()
        log("optimizer.step done on rank 1 only")
    log_weight_sums(model, "post-step (O04+O01):")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception:")
        traceback.print_exc()
        raise
