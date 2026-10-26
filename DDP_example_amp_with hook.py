"""N03 — fp16/bf16 compress hook: cast GradBucket before AllReduce.

Pattern (correct):
  1) take bucket.buffer() (usually fp32, often *GradScaler scale*)
  2) divide by world_size (mean)
  3) cast to fp16/bf16 for the collective  ← lossy / can saturate
  4) AllReduce on the narrow dtype
  5) cast back to original dtype in the Future callback for .grad

Risk: lost precision, bias, or Inf when |scaled grad| exceeds fp16 range (~65504).
bf16 has wider exponent (like fp32) so saturation is rarer; fp16 shows N03 harder.

Why one step often "looks the same": after GradScaler unscale, the weight update is
tiny; bf16/fp16 roundoff at .4f on weight_sum is invisible. Compare CAST=fp32 vs
fp16, or set AMPLIFY>1 so fp16 saturates to Inf (scaler then skips).

Env:
  CAST=bf16|fp16|fp32   (default fp16) — dtype used on the wire
  AMPLIFY=float         (default 1)    — multiply mean grads before cast (use 10 for Inf)
  MODE=ok|nan           (default ok)

Run (compare these two — look at reduced sum + weight_delta):
  CAST=fp32 AMPLIFY=1 torchrun --nproc_per_node=2 "DDP_example_amp_with hook.py"
  CAST=fp16 AMPLIFY=1 torchrun --nproc_per_node=2 "DDP_example_amp_with hook.py"
  CAST=fp16 AMPLIFY=10 torchrun --nproc_per_node=2 "DDP_example_amp_with hook.py"
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


def reduce_dtype_from_env() -> torch.dtype:
    name = os.environ.get("CAST", "bf16").lower()
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def grad_finite_report(model: DDP) -> tuple[bool, float, float]:
    total = 0.0
    max_abs = 0.0
    finite = True
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            finite = False
        total += g.sum().item()
        max_abs = max(max_abs, g.abs().max().item())
    return finite, total, max_abs


def DDP_comm_hook(process_group, bucket):
    """N03: mean in fp32, AllReduce in CAST dtype, return fp32 for DDP grads.

    Do not annotate bucket as dist.GradBucket under ``from __future__ import
    annotations`` — DDP compares the annotation object and stringized types fail.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    buf = bucket.buffer()
    reduce_dtype = reduce_dtype_from_env()

    log(
        f"comm_hook: recv bucket numel={buf.numel()} dtype={buf.dtype} "
        f"sum={buf.sum().item():.4f} max_abs={buf.abs().max().item():.4f} "
        f"finite={torch.isfinite(buf).all().item()}"
    )

    # Same math as default: mean = sum / world_size
    buf.div_(group.size())
    amplify = float(os.environ.get("AMPLIFY", "1"))
    if amplify != 1.0:
        buf.mul_(amplify)
        log(f"comm_hook: AMPLIFY={amplify} applied after mean (stress cast range)")

    orig_dtype = buf.dtype
    fp32_ref_sum = buf.float().sum().item()
    fp32_ref_max = buf.float().abs().max().item()

    cbuf = buf.to(reduce_dtype)
    # Quantization / saturation vs the fp32 mean we just computed
    cast_err = (cbuf.float() - buf.float()).abs().max().item()
    n_nonfinite = int((~torch.isfinite(cbuf)).sum().item())
    # Relative error on finite elements (makes small rounding visible)
    both_finite = torch.isfinite(cbuf) & torch.isfinite(buf)
    if both_finite.any():
        rel = (
            (cbuf.float() - buf.float()).abs()
            / buf.float().abs().clamp_min(1e-12)
        )[both_finite].max().item()
    else:
        rel = float("nan")
    log(
        f"comm_hook: N03 cast {orig_dtype} → {cbuf.dtype} | "
        f"fp32_pre_sum={fp32_ref_sum:.6f} max_abs={fp32_ref_max:.6f} | "
        f"cast max_abs={cbuf.float().abs().max().item():.6f} "
        f"max_abs_err={cast_err:.6g} max_rel_err={rel:.6g} nonfinite={n_nonfinite} "
        f"| fp16_max≈65504"
    )

    def _done(fut):
        reduced = fut.value()[0]
        out = reduced.to(orig_dtype)
        # Undo amplify so GradScaler unscale + step stay in a normal range when finite.
        if amplify != 1.0 and torch.isfinite(out).all():
            out = out / amplify
        delta_vs_local_fp32_mean = out.float().sum().item() - fp32_ref_sum
        # After AllReduce, sum should equal sum of per-rank means ≈ world mean*world
        # (we already divided by world_size, then SUM-allreduce → true mean)
        log(
            f"comm_hook: AFTER reduce wire={reduced.dtype} out={out.dtype} "
            f"sum={out.sum().item():.6f} max_abs={out.abs().max().item():.6f} "
            f"finite={torch.isfinite(out).all().item()} "
            f"sum_minus_local_fp32_pre={delta_vs_local_fp32_mean:.6f} "
            f"(ranks should MATCH each other on sum; CAST changes that value)"
        )
        return out

    return dist.all_reduce(cbuf, group=group, async_op=True).get_future().then(_done)


def main():
    setup()
    rank = dist.get_rank()
    mode = os.environ.get("MODE", "ok").lower()
    cast = os.environ.get("CAST", "fp16").lower()
    amplify = os.environ.get("AMPLIFY", "1")
    device = pick_device()
    use_cuda = device.type == "cuda"
    amp_device = "cuda" if use_cuda else "cpu"
    log(
        f"N03 MODE={mode} CAST={cast} AMPLIFY={amplify} device={device} "
        f"autocast={use_cuda} GradScaler(device={amp_device})"
    )

    torch.manual_seed(0)
    module = nn.Linear(10, 10).to(device)
    model = DDP(module, device_ids=[device.index] if use_cuda else None)
    model.register_comm_hook(state=dist.group.WORLD, hook=DDP_comm_hook)
    log(f"DDP + comm_hook registered (AllReduce in {reduce_dtype_from_env()})")

    w0 = model.module.weight.detach().float().sum().item()
    b0 = model.module.bias.detach().float().sum().item()
    log(f"init weight_sum={w0:.8f} bias_sum={b0:.8f}")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(amp_device, enabled=True)
    log(f"initial GradScaler scale={scaler.get_scale()} (scaled grads hit the hook)")

    torch.manual_seed(rank)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 10, device=device)

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(amp_device, enabled=use_cuda):
        loss = nn.functional.mse_loss(model(x), y)

    log(f"local loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")

    if mode == "nan" and rank == 1:
        loss = loss * float("nan")
        log("also N01: rank 1 set loss=NaN (mixes with N03)")
    elif mode not in ("ok", "nan"):
        raise SystemExit(f"unknown MODE={mode!r}; use ok|nan")

    log("calling scaler.scale(loss).backward() — hook runs inside backward")
    scaler.scale(loss).backward()

    finite, gsum, gmax = grad_finite_report(model)
    scale_now = scaler.get_scale()
    # .grad is still scaled until scaler.step unscales
    log(
        f"post-hook scaled .grad: finite={finite} sum={gsum:.6f} max_abs={gmax:.6f} "
        f"| approx unscaled_sum={gsum / scale_now:.8f} "
        f"(COMPARE this sum across CAST=fp32 vs CAST=fp16 runs)"
    )

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()
    log(
        f"GradScaler: scale {scale_before} → {scale_after} "
        f"(drop ⇒ Inf/NaN after lossy cast; step skipped)"
    )

    w1 = model.module.weight.detach().float().sum().item()
    b1 = model.module.bias.detach().float().sum().item()
    log(
        f"post-step weight_sum={w1:.8f} bias_sum={b1:.8f} | "
        f"delta_w={w1 - w0:.8e} delta_b={b1 - b0:.8e} "
        f"(fp32 vs fp16 runs: deltas differ if cast mattered)"
    )

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception:")
        traceback.print_exc()
        raise
