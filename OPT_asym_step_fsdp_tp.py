"""
Confirm: asymmetric optimizer.step / zero_grad under FSDP2 or TP.
STACK=fsdp|tp   MODE=healthy|double_step|skip_zero_grad

  Same lesson as DDP skip-step: step/zero_grad are local. Fwd/bwd collectives
  still match → no hang; local shards / weights diverge.

  double_step:     rank1 calls opt.step() twice on step 0.
  skip_zero_grad:  rank1 skips zero_grad on step 1 → stale grads add in.

  Terminal 0:
    STACK=fsdp MODE=double_step torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 OPT_asym_step_fsdp_tp.py
  Terminal 1:
    STACK=fsdp MODE=double_step torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 OPT_asym_step_fsdp_tp.py

  Then STACK=tp with the same MODE.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _wsum(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += _local(p).detach().float().sum().item()
    return s


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    stack = os.environ.get("STACK", "fsdp")
    mode = os.environ.get("MODE", "double_step")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    if stack == "fsdp":
        model = TinyLinear()
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        with torch.no_grad():
            model.lin.weight.fill_(0.01)
        model = model.to(device)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        in_dim = 16
    elif stack == "tp":
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
        in_dim = 8
    else:
        raise SystemExit(f"Unknown STACK={stack!r}; use fsdp|tp")

    opt = optim.SGD(model.parameters(), lr=0.1)
    w0 = _wsum(model)
    print(f"[rank {rank}] STACK={stack} MODE={mode} w0={w0:.4f}")

    for step in range(2):
        if mode == "skip_zero_grad" and rank == 1 and step == 1:
            print(f"[rank {rank}] step={step} skipping zero_grad")
        else:
            opt.zero_grad(set_to_none=True)

        x = torch.randn(4, in_dim, device=device)
        dist.broadcast(x, src=0)
        # Same x + TP plan → replicated loss already matches; no loss allreduce
        # (allreduce(loss) would warn: c10d is not autograd-differentiable).
        loss = model(x).float().sum()
        loss.backward()

        opt.step()
        if mode == "double_step" and rank == 1 and step == 0:
            opt.step()
            print(f"[rank {rank}] step={step} called optimizer.step() twice")

        print(
            f"[rank {rank}] step={step} loss={loss.item():.4f} "
            f"w={_wsum(model):.4f} delta_w={_wsum(model) - w0:.4f}"
        )

    # Collectives still fine after asymmetric local updates.
    dist.barrier()
    print(f"[rank {rank}] barrier_ok final_delta_w={_wsum(model) - w0:.4f}")
    if rank == 0:
        print(f"OK: opt asym demo finished (STACK={stack} MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
