"""
Different autograd hooks on TP DTensor params. MODE=healthy|diverge|rank1_only

  After parallelize_module, weights are DTensors (local shards). register_hook
  is per-process: each rank's hook runs only on that rank's backward.

  healthy:    both ranks scale fc1.grad by 0.5
  diverge:    rank0 ×0.5, rank1 ×2.0  → local shards update differently
  rank1_only: only rank1 registers a hook (×2); rank0 unmodified

  Watch: hook_fired, grad local_sum, delta_w (should match only in healthy),
         barrier_ok (no hang — hooks are local, not a collective).

  Terminal 0:
    MODE=diverge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 TP_hook_diverge.py
  Terminal 1:
    MODE=diverge torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 TP_hook_diverge.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "diverge")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))
    model = TinyMLP()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    with torch.no_grad():
        model.fc1.weight.fill_(0.01)
        model.fc2.weight.fill_(0.01)
    model = model.to(device)
    parallelize_module(
        model,
        mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )

    if mode == "healthy":
        scale = 0.5
        install = True
    elif mode == "diverge":
        scale = 0.5 if rank == 0 else 2.0
        install = True
    elif mode == "rank1_only":
        scale = 2.0
        install = rank == 1
    else:
        raise SystemExit(f"Unknown MODE={mode!r}; use healthy|diverge|rank1_only")

    fired: dict[str, int] = {"n": 0}

    def make_hook(name: str, s: float):
        def hook(grad: torch.Tensor):
            fired["n"] += 1
            g = _local(grad) if hasattr(grad, "to_local") else grad
            pre = g.detach().float().sum().item()
            out = grad * s
            post = _local(out).detach().float().sum().item()
            # pre can match across ranks (same math on different shards);
            # post = pre * scale is where ranks diverge under MODE=diverge.
            print(
                f"[rank {rank}] hook name={name} scale={s} "
                f"grad_type={type(grad).__name__} "
                f"local_grad_sum pre={pre:.4f} → post={post:.4f}"
            )
            return out

        return hook

    if install:
        # Hook the DTensor Parameter (the local shard lives inside it).
        model.fc1.weight.register_hook(make_hook("fc1.weight", scale))
        print(
            f"[rank {rank}] MODE={mode} registered fc1.weight hook scale={scale} "
            f"param_type={type(model.fc1.weight).__name__} "
            f"local_shape={tuple(_local(model.fc1.weight).shape)}"
        )
    else:
        print(f"[rank {rank}] MODE={mode} no hook installed")

    opt = optim.SGD(model.parameters(), lr=0.1)
    w0 = sum(_local(p).detach().float().sum().item() for p in model.parameters())

    opt.zero_grad(set_to_none=True)
    x = torch.randn(4, 8, device=device)
    dist.broadcast(x, src=0)
    loss = model(x).float().sum()
    loss.backward()

    g = model.fc1.weight.grad
    if g is None:
        gsum = float("nan")
    else:
        gsum = _local(g).detach().float().sum().item()
    print(
        f"[rank {rank}] after_bwd hook_fired={fired['n']} "
        f"fc1.grad_local_sum={gsum:.4f} (post-hook value on .grad)"
    )

    opt.step()
    w1 = sum(_local(p).detach().float().sum().item() for p in model.parameters())
    print(f"[rank {rank}] delta_w={w1 - w0:.4f}")

    dist.barrier()
    print(f"[rank {rank}] barrier_ok")
    if rank == 0:
        print(f"OK: TP hook diverge demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
