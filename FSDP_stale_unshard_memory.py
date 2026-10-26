"""
FSDP2 stale unshard memory probe. MODE=ref|realloc

  MODE=realloc torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=2950 FSDP_stale_unshard_memory.py
  # other terminal: node_rank=1, same MODE/port
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = os.environ.get("MODE", "realloc")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend=backend)
    torch.manual_seed(0)
    mesh = init_device_mesh(device, (world_size,))

    model = Tiny()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    # Distinct pattern so leftover bytes are recognizable.
    with torch.no_grad():
        model.lin.weight.copy_(torch.arange(64 * 64, dtype=torch.float32).view(64, 64))
    model = model.to(device)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)

    captured: dict = {}

    def _after_unshard(mod: nn.Module, _inputs: tuple) -> None:
        w = mod.lin.weight
        # After FSDP unshard this may be DTensor or a plain Tensor view.
        if isinstance(w, DTensor):
            full = w.full_tensor()
        else:
            full = w
        captured["full_sum"] = full.detach().float().sum().item()
        captured["full_shape"] = tuple(full.shape)
        captured["full_numel"] = full.numel()
        captured["full_dtype"] = full.dtype
        captured["device"] = full.device
        print(
            f"[rank {rank}] unshard window type={type(w).__name__} "
            f"full_sum={captured['full_sum']:.4f} shape={captured['full_shape']}"
        )
        if mode == "ref":
            # Keep a handle across reshard (may alias storage FSDP frees).
            captured["stale_ref"] = full
        else:
            captured["head"] = full.detach().float().flatten()[:8].cpu().clone()

    def _after_reshard(mod: nn.Module, _inputs: tuple, _output: torch.Tensor) -> None:
        # Runs after module forward; FSDP post-forward reshard should have run
        # if hooks are ordered: FSDP post hook registered first, then ours...
        # Register ours with prepend=False so we run AFTER FSDP's post_forward.
        print(f"[rank {rank}] post-forward hook (after module fwd)")

    # FSDP registers its hooks first; our pre-hook runs after unshard.
    model.register_forward_pre_hook(_after_unshard)
    # prepend=False (default): our post hook runs after earlier-registered FSDP post.
    model.register_forward_hook(_after_reshard)

    x = torch.randn(4, 64, device=device)
    dist.broadcast(x, src=0)
    y = model(x)
    print(f"[rank {rank}] out.sum={y.sum().item():.4f}")

    local = model.lin.weight.to_local()
    print(f"[rank {rank}] after reshard local_sum={local.sum().item():.4f}")

    if "full_shape" not in captured:
        print(f"[rank {rank}] WARNING: unshard capture missed (pre-hook did not record)")
    elif mode == "ref":
        stale = captured["stale_ref"]
        try:
            s = stale.detach().float().sum().item()
            print(f"[rank {rank}] stale_ref still readable sum={s:.4f}")
        except Exception as e:
            print(f"[rank {rank}] stale_ref read failed: {type(e).__name__}: {e}")
    elif mode == "realloc":
        # Allocate same-sized empty; allocator may reuse freed unshard buffer.
        buf = torch.empty(
            captured["full_shape"],
            dtype=captured["full_dtype"],
            device=captured["device"],
        )
        head = buf.detach().float().flatten()[:8].cpu()
        print(f"[rank {rank}] realloc head={head.tolist()}")
        print(f"[rank {rank}] unshard head={captured['head'].tolist()}")
        match = torch.equal(head, captured["head"])
        print(f"[rank {rank}] realloc_head_matches_unshard_head={match}")

    dist.barrier()
    if rank == 0:
        print(f"OK: stale-unshard demo finished (MODE={mode})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
