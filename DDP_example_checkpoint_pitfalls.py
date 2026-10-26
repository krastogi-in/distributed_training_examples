"""K01–K06 learning demos: DDP checkpoint pitfalls (CPU / gloo friendly).

These mirror how people implement save/load — useful for learning, not all are
“attacks.” DDP-specific freshness/skew: K02, K03. Common footguns: K05, K06.

Modes (env MODE=...):

  k06_bad   — save DDP state_dict (keys have ``module.``), load into bare nn.Module
              with strict=True → KeyError / missing keys. Default.
  k06_ok    — save ``model.module.state_dict()`` or strip ``module.`` → loads clean.
  k03_skew  — rank 0 saves step=1; rank 1 saves step=2 (no barrier / no agree);
              each loads its own file → skewed resume (log weight sums).
  k03_ok    — barrier; only rank 0 writes; all ranks load the same file after barrier.
  k02       — illustrate freshness: rank 0 saves early; rank 1 trains one more step
              then both load rank-0 ckpt → rank 1 rolls back (stale checkpoint).
  k05       — save on CPU; load with map_location that drops a key path mistake
              (demo: map_location="cpu" OK vs wrong dict handling).
  k01       — warn-only path: refuse to checkpoint if ``model.training`` mid-fake
              step flag set (document: never save mid-backward).

Run:
  torchrun --nproc_per_node=2 DDP_example_checkpoint_pitfalls.py
  MODE=k06_ok torchrun --nproc_per_node=2 DDP_example_checkpoint_pitfalls.py
  MODE=k03_skew torchrun --nproc_per_node=2 DDP_example_checkpoint_pitfalls.py
  MODE=k03_ok torchrun --nproc_per_node=2 DDP_example_checkpoint_pitfalls.py
  MODE=k02 torchrun --nproc_per_node=2 DDP_example_checkpoint_pitfalls.py

K04 (hash/signature on shared FS) is supply-chain policy — noted in logs only.
"""
from __future__ import annotations

import os
import socket
import traceback
from pathlib import Path

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
    log("process group ready")


def cleanup():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        log("destroyed process group")


def ckpt_dir() -> Path:
    d = Path(os.environ.get("CKPT_DIR", "/tmp/DDP_ckpt_demo"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def weight_sum(module: nn.Module) -> float:
    return sum(p.detach().float().sum().item() for p in module.parameters())


def one_step(model, optimizer, rank: int):
    torch.manual_seed(rank + 100)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def strip_module_prefix(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        out[k[len("module.") :] if k.startswith("module.") else k] = v
    return out


def mode_k06_bad(rank: int):
    """Save DDP-wrapped keys; load into bare module (strict) → fail or mismatch."""
    torch.manual_seed(0)
    bare = nn.Linear(10, 10)
    ddp = DDP(bare)
    opt = optim.SGD(ddp.parameters(), lr=0.05)
    one_step(ddp, opt, rank)
    path = ckpt_dir() / "k06_ddp.pt"
    if rank == 0:
        # Common mistake: torch.save(ddp.state_dict())  → keys like module.weight
        torch.save({"model": ddp.state_dict(), "step": 1}, path)
        log(f"K06 bad: saved DDP keys sample={list(ddp.state_dict().keys())[:2]}")
    dist.barrier()

    fresh = nn.Linear(10, 10)  # bare, no DDP
    blob = torch.load(path, map_location="cpu", weights_only=False)
    log(f"K06 bad: loading into bare module; ckpt keys={list(blob['model'].keys())[:2]}")
    try:
        fresh.load_state_dict(blob["model"], strict=True)
        log("K06 unexpected: strict load succeeded")
    except Exception as e:
        log(f"K06 expected failure (module. prefix): {type(e).__name__}: {e}")


def mode_k06_ok(rank: int):
    torch.manual_seed(0)
    bare = nn.Linear(10, 10)
    ddp = DDP(bare)
    opt = optim.SGD(ddp.parameters(), lr=0.05)
    one_step(ddp, opt, rank)
    path = ckpt_dir() / "k06_ok.pt"
    if rank == 0:
        # Correct: save bare module (or strip prefix)
        torch.save({"model": ddp.module.state_dict(), "step": 1}, path)
        log(f"K06 ok: saved bare keys sample={list(ddp.module.state_dict().keys())[:2]}")
    dist.barrier()
    fresh = nn.Linear(10, 10)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    fresh.load_state_dict(blob["model"], strict=True)
    log(f"K06 ok: loaded; weight_sum={weight_sum(fresh):.6f}")


def mode_k03_skew(rank: int):
    """Each rank writes/loads its own file — no barrier / no single source of truth.

    Both ranks must take the same number of DDP steps (collectives). Skew comes from
    per-rank checkpoints + a local-only tweak on rank 1 before save.
    """
    torch.manual_seed(0)
    bare = nn.Linear(10, 10)
    ddp = DDP(bare)
    opt = optim.SGD(ddp.parameters(), lr=0.05)
    for s in range(2):
        one_step(ddp, opt, rank)
        log(f"K03 skew: DDP step {s + 1}/2 w={weight_sum(ddp.module):.6f}")

    claimed_step = 1 if rank == 0 else 2
    if rank == 1:
        # Local alteration after training (no AllReduce) — simulates divergent resume source.
        with torch.no_grad():
            ddp.module.bias.add_(1.0)
        log("K03 skew: rank1 locally altered bias before its private save")

    path = ckpt_dir() / f"k03_rank{rank}.pt"
    torch.save(
        {"model": ddp.module.state_dict(), "step": claimed_step, "rank": rank},
        path,
    )
    log(f"K03 skew: saved {path} claimed_step={claimed_step} (no shared barrier/agree)")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    ddp.module.load_state_dict(blob["model"])
    log(
        f"K03 skew: resumed claimed_step={blob['step']} "
        f"weight_sum={weight_sum(ddp.module):.6f} (compare across ranks — divergent)"
    )


def mode_k03_ok(rank: int):
    torch.manual_seed(0)
    bare = nn.Linear(10, 10)
    ddp = DDP(bare)
    opt = optim.SGD(ddp.parameters(), lr=0.05)
    one_step(ddp, opt, rank)
    dist.barrier()  # all ranks finished the step
    path = ckpt_dir() / "k03_shared.pt"
    if rank == 0:
        torch.save({"model": ddp.module.state_dict(), "step": 1}, path)
        log(f"K03 ok: rank0 wrote shared ckpt step=1 w={weight_sum(ddp.module):.6f}")
    dist.barrier()  # do not load until write visible
    blob = torch.load(path, map_location="cpu", weights_only=False)
    ddp.module.load_state_dict(blob["model"])
    log(
        f"K03 ok: all ranks loaded step={blob['step']} "
        f"weight_sum={weight_sum(ddp.module):.6f} (should match)"
    )


def mode_k02(rank: int):
    """Stale checkpoint freshness: save after step1, train step2, then all load step1."""
    torch.manual_seed(0)
    bare = nn.Linear(10, 10)
    ddp = DDP(bare)
    opt = optim.SGD(ddp.parameters(), lr=0.05)
    one_step(ddp, opt, rank)  # step 1 — both ranks (required for DDP)
    dist.barrier()
    path = ckpt_dir() / "k02_stale.pt"
    if rank == 0:
        torch.save({"model": ddp.module.state_dict(), "step": 1}, path)
        log(f"K02: rank0 saved step1 w={weight_sum(ddp.module):.6f}")
    dist.barrier()

    one_step(ddp, opt, rank)  # step 2 — both advance; ckpt is now stale
    log(f"K02: after step2 (live) w={weight_sum(ddp.module):.6f}")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    before = weight_sum(ddp.module)
    ddp.module.load_state_dict(blob["model"])
    after = weight_sum(ddp.module)
    log(
        f"K02: loaded STALE step={blob['step']} before={before:.6f} after={after:.6f} "
        f"(rolled back from step2 → step1)"
    )


def mode_k05(rank: int):
    """map_location: correct CPU remap vs forgetting it when devices differ."""
    torch.manual_seed(0)
    m = nn.Linear(10, 10)
    path = ckpt_dir() / "k05.pt"
    if rank == 0:
        torch.save(m.state_dict(), path)
        log("K05: saved CPU state_dict")
    dist.barrier()

    # OK on this gloo CPU demo
    ok = torch.load(path, map_location="cpu", weights_only=True)
    m2 = nn.Linear(10, 10)
    m2.load_state_dict(ok)
    log(f"K05 ok: map_location='cpu' weight_sum={weight_sum(m2):.6f}")

    # Footgun pattern people hit on GPU boxes: load without map_location after
    # save from cuda:0 onto a rank that only has cuda:1 / cpu → error or wrong device.
    log(
        "K05 note: on multi-GPU, always torch.load(..., map_location=f'cuda:{local_rank}') "
        "or map_location='cpu' then .to(device). Skipping map_location is the footgun."
    )


def mode_k01(rank: int):
    """Document: never checkpoint mid-backward / mid-optimizer step."""
    log(
        "K01: saving mid-backward or mid-scaler.step can capture half-updated "
        "params / missing grads / optimizer state mid-mutation → non-global step."
    )
    log(
        "K01 rule: checkpoint only at a quiet point — after optimizer.step "
        "(and scaler.update), before next forward; barrier so all ranks agree."
    )
    in_backward = True  # pretend
    if in_backward:
        log("K01 demo: refusing save because in_backward=True (app-level guard)")
    dist.barrier()


def main():
    setup()
    rank = dist.get_rank()
    mode = os.environ.get("MODE", "k06_bad").lower()
    log(f"checkpoint pitfalls MODE={mode} dir={ckpt_dir()}")
    log(
        "K04 (supply chain): on shared FS, verify checksum/signature before load — "
        "not demoed; policy/control, not a DDP API."
    )

    {
        "k06_bad": mode_k06_bad,
        "k06_ok": mode_k06_ok,
        "k03_skew": mode_k03_skew,
        "k03_ok": mode_k03_ok,
        "k02": mode_k02,
        "k05": mode_k05,
        "k01": mode_k01,
    }[mode](rank)

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        log(f"unknown MODE; use k06_bad|k06_ok|k03_skew|k03_ok|k02|k05|k01 ({e})")
        raise
    except Exception:
        log("FAILED:")
        traceback.print_exc()
        raise
