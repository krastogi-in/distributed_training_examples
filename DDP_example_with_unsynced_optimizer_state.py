"""DDP: O03 — optimizer state is local; DDP does not AllReduce Adam moments.

DDP only synchronizes gradients during backward. Each rank keeps its own
optimizer.state (e.g. Adam exp_avg / exp_avg_sq, SGD momentum buffers).
With symmetric training they stay aligned; any asymmetry in updates or state
makes effective steps differ even when post-backward .grad matches.

Run:
  torchrun --nproc_per_node=2 DDP_example_with_unsynced_optimizer_state.py
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


def DDP_comm_hook(process_group, bucket: dist.GradBucket):
    group = process_group if process_group is not None else dist.group.WORLD
    buf = bucket.buffer()
    buf.div_(group.size())

    def _done(fut):
        return fut.value()[0]

    return dist.all_reduce(buf, group=group, async_op=True).get_future().then(_done)


def adam_moment_sums(optimizer: optim.Optimizer) -> tuple[float, float]:
    exp_avg = 0.0
    exp_avg_sq = 0.0
    for state in optimizer.state.values():
        if "exp_avg" in state:
            exp_avg += state["exp_avg"].sum().item()
        if "exp_avg_sq" in state:
            exp_avg_sq += state["exp_avg_sq"].sum().item()
    return exp_avg, exp_avg_sq


def train_step(model, optimizer, x, y, world: int):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    log(f"local loss before sync={loss.item():.4f} (may differ per rank)")
    loss.backward()
    log(f"backward done — grads allreduced over world_size={world}")
    for name, param in model.module.named_parameters():
        g = param.grad.sum().item() if param.grad is not None else float("nan")
        log(f"post-backward {name}.grad_sum={g:.4f} (should match across ranks)")


def main():
    setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    torch.manual_seed(0)
    module = nn.Linear(10, 10)
    model = DDP(module)
    model.register_comm_hook(state=dist.group.WORLD, hook=DDP_comm_hook)
    log("DDP wrapped model + comm_hook registered")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    torch.manual_seed(rank)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    log("--- step 1: symmetric Adam.step → optimizer state should match ---")
    train_step(model, optimizer, x, y, world)
    optimizer.step()
    m1, v1 = adam_moment_sums(optimizer)
    log(f"after step 1 Adam moments: exp_avg_sum={m1:.4f} exp_avg_sq_sum={v1:.4f}")

    dist.barrier()
    if rank == 0:
        log("(rank 0) moments should equal rank 1 after a symmetric step")

    log("--- step 2: same synced grad, but rank 1 corrupts local Adam state ---")
    train_step(model, optimizer, x, y, world)
    m_before, v_before = adam_moment_sums(optimizer)
    log(f"pre-corruption moments: exp_avg_sum={m_before:.4f} exp_avg_sq_sum={v_before:.4f}")

    if rank == 1:
        for state in optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].mul_(10.0)
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(10.0)
        log("O03: rank 1 scaled exp_avg and exp_avg_sq by 10 (DDP does not sync this)")

    m_after, v_after = adam_moment_sums(optimizer)
    log(f"pre-step moments: exp_avg_sum={m_after:.4f} exp_avg_sq_sum={v_after:.4f}")

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
