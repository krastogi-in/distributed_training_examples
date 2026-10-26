"""
Inf in FSDP weights vs Inf grads — reshard still runs; training stalls.
MODE=inf_grad|inf_weight

  Differs from AMP_found_inf_max_reduce.py:
    That probe: Inf grads → found_inf MAX → scaler skip (plain Linear).
    This probe: same skip under FSDP2, plus Inf *in the param shard*, and an
    explicit unshard/reshard after the bad step to show collectives still finish.

  inf_grad:   rank1 poisons grads each step → skip; weights stay finite; reshard ok.
  inf_weight: rank1 poisons param shard once → Inf survives in weights; loss/grads
              stay non-finite; scaler keeps skipping; reshard still ok (no hang).

  Terminal 0:
    MODE=inf_weight torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 AMP_inf_weight_fsdp_reshard.py
  Terminal 1:
    MODE=inf_weight torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 AMP_inf_weight_fsdp_reshard.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.lin(x)


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "inf_weight")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))
    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    with torch.no_grad():
        model.lin.weight.fill_(0.01)
    model = model.to(device)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)

    opt = optim.SGD(model.parameters(), lr=0.1)
    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=True)

    def wsum() -> float:
        return _local(model.lin.weight).detach().float().sum().item()

    def w_finite() -> bool:
        return bool(torch.isfinite(_local(model.lin.weight)).all())

    w0 = wsum()
    print(f"[rank {rank}] MODE={mode} w0={w0:.4f} w_finite={w_finite()}")

    if mode == "inf_weight" and rank == 1:
        with torch.no_grad():
            _local(model.lin.weight).fill_(float("inf"))
        print(f"[rank {rank}] poisoned weight shard (Inf lives in params)")

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, 16, device=device)
        dist.broadcast(x, src=0)

        with torch.autocast(device_type=amp_device, enabled=(device == "cuda")):
            loss = model(x).float().sum()
        print(
            f"[rank {rank}] step={step} forward_ok loss_finite="
            f"{bool(torch.isfinite(loss))} w_finite={w_finite()}"
        )

        scaler.scale(loss).backward()

        if mode == "inf_grad" and rank == 1:
            for p in model.parameters():
                if p.grad is not None:
                    _local(p.grad).data.fill_(float("inf"))
            print(f"[rank {rank}] step={step} poisoned grads only")

        found = torch.zeros(1, device=device)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(_local(p.grad)).all():
                found.fill_(1.0)
                break
        dist.all_reduce(found, op=dist.ReduceOp.MAX)
        global_found = int(found.item())
        if global_found:
            for p in model.parameters():
                if p.grad is not None:
                    _local(p.grad).data.fill_(float("nan"))

        scale_b = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        scale_a = scaler.get_scale()

        # Claim under test: Inf does not break FSDP allgather/reduce-scatter.
        model.unshard()
        model.reshard()
        print(
            f"[rank {rank}] step={step} global_found_inf={global_found} "
            f"scale {scale_b}→{scale_a} delta_w={wsum() - w0:.4f} "
            f"w_finite={w_finite()} unshard/reshard=ok"
        )

    dist.barrier()
    if rank == 0:
        print(f"OK: Inf-weight FSDP reshard demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
