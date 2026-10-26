"""N04 — DDP mixed precision vs AMP: two stacks / unexpected dtype on wire.

Quick check modes (env MODE=...):

  mismatch  — rank 0 keeps fp32 grads; rank 1 casts bucket to fp16 before
              AllReduce → collective dtype mismatch → ERROR (default).
  stack     — both ranks use GradScaler AMP *and* cast wire to fp16 in the
              hook (two stacks, same on every rank) → runs; log wire dtype
              + cast error (distortion, not necessarily a crash).
  amp_only  — GradScaler + default fp32 AllReduce (baseline).

Run:
  torchrun --nproc_per_node=2 DDP_example_amp_mixed_precision_mismatch.py
  MODE=stack torchrun --nproc_per_node=2 DDP_example_amp_mixed_precision_mismatch.py
  MODE=amp_only torchrun --nproc_per_node=2 DDP_example_amp_mixed_precision_mismatch.py
"""
from __future__ import annotations

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
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    log(f"joining process group via {info['master']} (backend={backend})")
    dist.init_process_group(backend=backend)


def cleanup():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        log("destroyed process group")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return torch.device("cuda", local)
    return torch.device("cpu")


def make_comm_hook(mode: str):
    """Hook chosen by MODE — no GradBucket annotation (future annotations)."""

    def hook(process_group, bucket):
        group = process_group if process_group is not None else dist.group.WORLD
        buf = bucket.buffer()
        buf.div_(group.size())
        orig = buf.dtype
        rank = group.rank()

        if mode == "amp_only":
            wire = buf
            log(f"comm_hook amp_only: wire dtype={wire.dtype} (fp32 baseline)")
        elif mode == "stack":
            # Stack 2: AMP already scaled grads; we also compress the wire.
            wire = buf.to(torch.float16)
            err = (wire.float() - buf.float()).abs().max().item()
            log(
                f"comm_hook stack: AMP+cast wire={wire.dtype} "
                f"max_cast_err={err:.6g} (same on all ranks → distortion OK for collective)"
            )
        elif mode == "mismatch":
            # Rank 0: fp32 wire; rank 1: fp16 wire → dtype mismatch.
            if rank == 1:
                wire = buf.to(torch.float16)
                log(f"comm_hook mismatch: rank1 wire dtype={wire.dtype}")
            else:
                wire = buf
                log(f"comm_hook mismatch: rank0 wire dtype={wire.dtype}")
        else:
            raise RuntimeError(f"unknown MODE={mode}")

        def _done(fut):
            out = fut.value()[0].to(orig)
            log(
                f"comm_hook done: out dtype={out.dtype} sum={out.sum().item():.6f} "
                f"finite={torch.isfinite(out).all().item()}"
            )
            return out

        return dist.all_reduce(wire, group=group, async_op=True).get_future().then(_done)

    return hook


def main():
    setup()
    rank = dist.get_rank()
    mode = os.environ.get("MODE", "mismatch").lower()
    device = pick_device()
    use_cuda = device.type == "cuda"
    amp_device = "cuda" if use_cuda else "cpu"
    log(f"N04 MODE={mode} device={device}")

    torch.manual_seed(0)
    module = nn.Linear(10, 10).to(device)
    # Stack 1: keep master weights fp32; AMP via GradScaler (and autocast on CUDA).
    model = DDP(module, device_ids=[device.index] if use_cuda else None)
    model.register_comm_hook(state=dist.group.WORLD, hook=make_comm_hook(mode))
    log("DDP + mode-specific comm_hook registered")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(amp_device, enabled=True)
    log(f"GradScaler scale={scaler.get_scale()} (AMP stack always on)")

    torch.manual_seed(rank)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 10, device=device)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(amp_device, enabled=use_cuda):
        loss = nn.functional.mse_loss(model(x), y)
    log(f"loss={loss.item():.4f}")

    log("backward (AllReduce dtype decided inside hook)…")
    scaler.scale(loss).backward()

    gsum = sum(
        p.grad.float().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    log(f"post-backward grad_sum={gsum:.6f}")

    scaler.step(optimizer)
    scaler.update()
    log(f"done weight_sum={model.module.weight.detach().sum().item():.6f}")
    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED (expected for MODE=mismatch):")
        traceback.print_exc()
        raise
