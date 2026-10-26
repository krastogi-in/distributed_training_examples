"""
AMP found_inf MAX-reduce probe. MODE=healthy|force_skip

  What:
    Each rank checks whether grads are finite (found_inf 0/1). Distributed AMP
    does ALLREDUCE(MAX) on that flag. If ANY rank has Inf/NaN, MAX=1 everywhere
    → every rank skips optimizer.step() that iteration.

    force_skip: rank1 injects Inf grads every step → global skip forever
                even though rank0's loss/grads were finite.

  MODE=force_skip torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 AMP_found_inf_max_reduce.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


def main() -> None:
    rank = int(os.environ["RANK"])
    mode = os.environ.get("MODE", "force_skip")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(0)
    model = nn.Linear(16, 16, bias=False).to(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    opt = optim.SGD(model.parameters(), lr=0.1)
    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=True)

    w0 = model.weight.detach().sum().item()
    print(f"[rank {rank}] MODE={mode} scale0={scaler.get_scale()} w_sum0={w0:.4f}")

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, 16, device=device)
        dist.broadcast(x, src=0)

        with torch.autocast(device_type=amp_device, enabled=(device == "cuda")):
            loss = model(x).float().sum()

        scaler.scale(loss).backward()

        if mode == "force_skip" and rank == 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("inf"))
            print(f"[rank {rank}] step={step} injected Inf grads (local)")

        # --- found_inf MAX-reduce (the distributed AMP contract) ---
        found = torch.zeros(1, device=device)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                found.fill_(1.0)
                break
        local_found = int(found.item())
        dist.all_reduce(found, op=dist.ReduceOp.MAX)
        global_found = int(found.item())

        # If anyone saw Inf, make EVERY rank's grads non-finite so GradScaler
        # skips the step on all ranks (same outcome as synced found_inf).
        if global_found:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("nan"))

        scale_before = scaler.get_scale()
        scaler.step(opt)  # skips when non-finite grads
        scaler.update()
        scale_after = scaler.get_scale()

        w = model.weight.detach().sum().item()
        print(
            f"[rank {rank}] step={step} local_found={local_found} "
            f"global_found_inf={global_found} "
            f"scale {scale_before}→{scale_after} w_sum={w:.4f}"
        )

    w1 = model.weight.detach().sum().item()
    print(f"[rank {rank}] delta_w_sum={w1 - w0:.4f}")

    dist.barrier()
    if rank == 0:
        print(f"OK: found_inf MAX demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
