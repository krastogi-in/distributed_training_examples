"""N01 — One rank Inf/NaN under AMP: Scaler + AllReduce interaction.

AMP in one process (no DDP)
---------------------------
  with autocast(...):          # forward in fp16/bf16 (faster, less memory)
      loss = ...
  scaler.scale(loss).backward()  # multiply loss by scale so tiny fp16 grads survive
  scaler.step(optimizer)         # unscale grads; if Inf/NaN → SKIP optimizer.step
  scaler.update()                # if Inf/NaN seen → shrink scale; else maybe grow it

DDP + AMP
---------
  DDP AllReduces grads during backward (after scale(loss).backward starts).
  GradScaler state (the scale factor) is LOCAL — not AllReduced (see N02).

N01 risk
--------
  One rank produces Inf/NaN grads (overflow, bad batch, injected NaN).
  AllReduce(SUM/mean) with any NaN → NaN on EVERY rank.
  Then every rank's GradScaler sees found_inf → skip step + backoff scale.
  Training can stall on a permanently tiny scale / repeated skips ("global NaN").

Modes (env MODE=...):
  ok   — clean step; grads finite; scaler may keep/grow scale
  nan  — rank 1 sets loss to NaN before backward (default) → all ranks get NaN grads

Run (CUDA preferred for real GradScaler; CPU still shows AllReduce NaN spread):
  torchrun --nproc_per_node=2 DDP_example_amp_nan_rank.py
  MODE=ok torchrun --nproc_per_node=2 DDP_example_amp_nan_rank.py
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
    # Prefer gloo on CPU; nccl if every rank has a CUDA device.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    log(f"joining process group via {info['master']} (backend={backend})")
    dist.init_process_group(backend=backend)
    log("process group ready — this rank is in the job")


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        log("destroyed process group")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return torch.device("cuda", local)
    return torch.device("cpu")


def grad_finite_report(model: DDP) -> tuple[bool, float]:
    total = 0.0
    finite = True
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            finite = False
        total += p.grad.float().sum().item()
    return finite, total


def main():
    setup()
    rank = dist.get_rank()
    mode = os.environ.get("MODE", "nan").lower()
    device = pick_device()
    use_cuda = device.type == "cuda"
    amp_device = "cuda" if use_cuda else "cpu"
    # autocast fp16 is mainly useful on CUDA; keep GradScaler enabled on CPU too so
    # Inf/NaN → skip step + backoff still demonstrates N01 without a GPU.
    log(f"N01 MODE={mode} device={device} autocast={use_cuda} GradScaler(device={amp_device})")

    torch.manual_seed(0)
    module = nn.Linear(10, 10).to(device)
    model = DDP(module, device_ids=[device.index] if use_cuda else None)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    scaler = torch.amp.GradScaler(amp_device, enabled=True)
    log(f"initial GradScaler scale={scaler.get_scale()}")

    torch.manual_seed(rank)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 10, device=device)

    optimizer.zero_grad(set_to_none=True)

    # --- AMP forward ---
    # autocast: matmuls etc. run in lower precision when CUDA AMP is on.
    with torch.amp.autocast(amp_device, enabled=use_cuda):
        loss = nn.functional.mse_loss(model(x), y)

    log(f"local loss (pre-inject)={loss.item():.4f}")

    if mode == "nan" and rank == 1:
        # Simulate overflow / corrupt batch on ONE rank only.
        loss = loss * float("nan")
        log("N01: rank 1 replaced loss with NaN (before scale+backward)")
    elif mode != "ok" and mode != "nan":
        raise SystemExit(f"unknown MODE={mode!r}; use ok|nan")

    log(f"loss going into scaler.scale: finite={torch.isfinite(loss).item()}")

    # --- scale + backward (DDP AllReduce runs inside backward) ---
    # scale(loss) = loss * scale_factor; grads are also scaled until unscale_/step.
    scaler.scale(loss).backward()

    finite, gsum = grad_finite_report(model)
    log(
        f"after AllReduce: grads_finite={finite} grad_sum={gsum:.4f} "
        f"(NaN on one rank → NaN everywhere after reduce)"
    )

    scale_before = scaler.get_scale()
    # step: unscale, then optimizer.step OR skip if Inf/NaN found
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    log(
        f"GradScaler: scale {scale_before} → {scale_after} "
        f"(shrink means found Inf/NaN and skipped optimizer.step)"
    )

    w_sum = model.module.weight.detach().float().sum().item()
    log(f"post-step weight_sum={w_sum:.4f}")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception:")
        traceback.print_exc()
        raise
