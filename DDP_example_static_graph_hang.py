"""H04 — Lie about static_graph / unused params → hang or graph-change error.

DDP schedules gradient AllReduce assuming every rank marks the same buckets
"ready". That assumption comes from config:

  find_unused_parameters=False  (default): every param gets a grad every step
  static_graph=True:            used/unused set is fixed after the first iters

If the real graph violates that (optional head, conditional loss, rank skew),
reduction never completes on some ranks → hang (DoS-shaped) or RuntimeError.

Modes (env MODE=...):
  unused  — find_unused_parameters=False; rank 0 uses both heads, rank 1 skips
            head_b → AllReduce hang (rank 0 stuck; rank 1 may finish). Default.
  static  — static_graph=True; iter 0 leaves head_b unused, iter 1 uses it
            → RuntimeError: training graph has changed.
  fixed   — find_unused_parameters=True; same skew as unused → completes.

Run:
  timeout 15 torchrun --nproc_per_node=2 DDP_example_static_graph_hang.py
  MODE=static torchrun --nproc_per_node=2 DDP_example_static_graph_hang.py
  MODE=fixed  torchrun --nproc_per_node=2 DDP_example_static_graph_hang.py
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
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        log("destroyed process group")


class TwoHead(nn.Module):
    """shared → head_a always; head_b optional (unused when use_b=False)."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 10)
        self.head_a = nn.Linear(10, 10)
        self.head_b = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor, use_b: bool) -> torch.Tensor:
        h = self.shared(x)
        out = self.head_a(h)
        if use_b:
            out = out + self.head_b(h)
        return out


def one_step(model, optimizer, x, y, use_b: bool, label: str):
    optimizer.zero_grad()
    log(f"{label}: use_b={use_b}")
    loss = nn.functional.mse_loss(model(x, use_b=use_b), y)
    log(f"{label}: local loss={loss.item():.4f} — entering backward (collective)")
    loss.backward()
    log(f"{label}: backward done")
    optimizer.step()


def main():
    setup()
    rank = dist.get_rank()
    mode = os.environ.get("MODE", "unused").lower()
    log(f"H04 MODE={mode}")

    torch.manual_seed(0)
    module = TwoHead()

    if mode == "fixed":
        model = DDP(module, find_unused_parameters=True)
        log("DDP(find_unused_parameters=True) — unused head_b is OK")
    elif mode == "static":
        model = DDP(module, static_graph=True)
        log("DDP(static_graph=True) — used/unused set must not change after warm-up")
    elif mode == "unused":
        model = DDP(module, find_unused_parameters=False)
        log(
            "DDP(find_unused_parameters=False) — every param must get a grad; "
            "asymmetric use_b will hang in AllReduce"
        )
    else:
        raise SystemExit(f"unknown MODE={mode!r}; use unused|static|fixed")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    if mode == "static":
        # Record head_b as unused, then use it — classic static_graph violation.
        # (used→unused often still "works" on current PyTorch; unused→used errors.)
        one_step(model, optimizer, x, y, use_b=False, label="iter0")
        dist.barrier()
        log("H04: enabling head_b (was unused) — violates static_graph")
        one_step(model, optimizer, x, y, use_b=True, label="iter1")
        log("unexpected: static mode completed without error")
    else:
        # unused / fixed: rank 0 uses both heads; rank 1 skips head_b.
        use_b = rank == 0
        log(
            f"H04: rank-asymmetric graph — rank0 use_b=True, rank1 use_b=False "
            f"(this rank use_b={use_b})"
        )
        one_step(model, optimizer, x, y, use_b=use_b, label="step0")
        if mode == "fixed":
            log("fixed: completed — find_unused_parameters=True handled unused head_b")
        else:
            # Rank that skipped head_b may finish backward; the peer hangs in AllReduce.
            log(
                "local backward returned — if peer still has head_b grads to reduce, "
                "the job hangs here (use timeout). Peer stuck = H04 DoS."
            )
            dist.barrier()
            log("unexpected: unused mode passed barrier (expected hang)")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED with exception (expected for MODE=static):")
        traceback.print_exc()
        raise
