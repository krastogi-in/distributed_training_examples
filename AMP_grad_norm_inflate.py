"""
Distributed clip_grad_norm_: inflate local norm vs skip the allreduce.
MODE=healthy|inflate|skip_allreduce

  Correction vs "this will halt":
    inflate:        every rank still allreduces sq-norm. Rank1 multiplies its
                    local contribution before the reduce → global total_norm too
                    big → clip_coef too small → everyone's grads shrink.
                    Wrong math, finishes fine (silent wrong).
    skip_allreduce: rank1 skips the allreduce → hang / NCCL timeout (true stall).

  Manual clip (same contract as torch.nn.utils.clip_grad_norm_ + DDP/FSDP):
    total_norm = sqrt(allreduce_sum(sum(g^2)))
    coef = min(1, max_norm / (total_norm + eps)); g *= coef

  Terminal 0:
    MODE=inflate torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 AMP_grad_norm_inflate.py
  Terminal 1:
    MODE=inflate torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
      --master_addr=127.0.0.1 --master_port=2950 AMP_grad_norm_inflate.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


MAX_NORM = 1.0
INFLATE = 100.0  # rank1 lies: pretends local sq-norm is INFLATE× larger


def main() -> None:
    rank = int(os.environ["RANK"])
    mode = os.environ.get("MODE", "inflate")
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
    w0 = model.weight.detach().sum().item()
    print(f"[rank {rank}] MODE={mode} max_norm={MAX_NORM} w0={w0:.4f}")

    for step in range(2):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, 16, device=device)
        dist.broadcast(x, src=0)
        loss = model(x).sum()
        loss.backward()

        # Local sum of squares (before any lie).
        local_sq = torch.zeros(1, device=device)
        for p in model.parameters():
            if p.grad is not None:
                local_sq += p.grad.detach().float().pow(2).sum()
        honest_local_sq = local_sq.item()

        contrib = local_sq.clone()
        if mode == "inflate" and rank == 1:
            contrib *= INFLATE
            print(
                f"[rank {rank}] step={step} lying: local_sq "
                f"{honest_local_sq:.4f} → contrib {contrib.item():.4f}"
            )

        if mode == "skip_allreduce" and rank == 1:
            print(f"[rank {rank}] step={step} skipping norm allreduce (expect hang)")
            # Rank0 will block in all_reduce below.
        else:
            dist.all_reduce(contrib, op=dist.ReduceOp.SUM)

        total_norm = contrib.sqrt().item()
        coef = min(1.0, MAX_NORM / (total_norm + 1e-6))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(coef)

        g_after = sum(
            p.grad.detach().float().norm().item()
            for p in model.parameters()
            if p.grad is not None
        )
        opt.step()
        w = model.weight.detach().sum().item()
        print(
            f"[rank {rank}] step={step} honest_local_sq={honest_local_sq:.4f} "
            f"total_norm={total_norm:.4f} coef={coef:.6f} "
            f"grad_l2_after={g_after:.4f} delta_w={w - w0:.4f}"
        )

    dist.barrier()
    if rank == 0:
        print(f"OK: grad-norm demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
