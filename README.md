# PyTorch Distributed Assessment Lab

Hands-on probes of **`torch.distributed`** and related stacks (DDP, FSDP2, DTensor, Tensor/Context Parallel, Distributed Checkpoint, RPC, process groups). The goal is to map **what can go wrong** when ranks disagree, a peer misbehaves, a checkpoint store is untrusted, or ops configuration drifts — covering **security**, **correctness**, **hangs**, and **silent poisoning**, not only “happy path” tutorials.

> **Lab only.** These scripts intentionally create divergence, tampering, and unsafe deserialization paths against **your own** local jobs. Do not point them at production clusters or other people’s stores.

---

## Trust model (why these matter)

PyTorch distributed largely assumes:

| Assumption | If broken… |
|------------|------------|
| All ranks are **trusted** and run the same intended program | One rank can poison grads, LSE, shards, hooks, or optimizer steps |
| **TCPStore / rendezvous** is reachable only by job members | Any client that can reach `host:port` can read/write/delete keys (**no auth**) |
| Checkpoint **store is trusted** | DCP has no built-in checksum/signature; pickle/`torch.load` surfaces exist |
| Ranks agree on **backend, mesh, plan, dtype, timeouts** | Hangs, loud errors, or **silent** wrong results |
| Collectives are **collective** (same order, same participation) | Deadlock or desync that looks like “training instability” |

Many failures are **not remote CVEs**; they are **trust-boundary / ops / SPMD contract** issues. That is still in scope: multi-tenant or compromised-worker settings turn them into real risk.

---

## Requirements

- Python 3.10+ recommended  
- Recent PyTorch (probes were developed against ~2.x; DCP / FSDP2 / CP APIs move — expect version sensitivity)  
- `torchrun` (ships with PyTorch)  
- Most scripts run on **CPU + gloo**; some prefer **CUDA + nccl** when available  

```bash
# example
pip install torch
# or use your conda env (scripts sometimes mention `cuda_test`)
```

---

## How to run

### Single machine, N processes (simple)

```bash
torchrun --nproc_per_node=2 DDP_example_1.py
```

### Two-terminal “2 nodes × 1 proc” pattern (most probes)

Many scripts expect **two shells** with the same `MODE` / port:

```bash
# terminal 0
MODE=… torchrun --nnodes=2 --nproc-per-node=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=2950 SCRIPT.py

# terminal 1
MODE=… torchrun --nnodes=2 --nproc-per-node=1 --node_rank=1 \
  --master_addr=127.0.0.1 --master_port=2950 SCRIPT.py
```

### `MODE` and other knobs

Most probes select behavior with env vars, e.g. `MODE=healthy|…`. See each file’s module docstring for exact values. Common outcomes called out in comments:

- **ok** — expected healthy behavior  
- **loud_error** — exception / timeout (detectable)  
- **silent_poison** — continues with wrong weights / attention / checkpoint (worst class)  
- **hang** — collective / init deadlock  

Single-process audits (no `torchrun`):

```bash
python DCP_pickle_surface_audit.py
MODE=bitflip python DCP_chunk_tamper_integrity.py
```

---

## Naming map

| Prefix | Area |
|--------|------|
| `DDP_example_*` | DistributedDataParallel + hooks / AMP / checkpoints |
| `DTensor_*` / `DTensor_*` | DTensor mesh, placements, redistribute, SPMD |
| `FSDP_*` | FSDP2 (`fully_shard`) lifecycle / sync |
| `TP_*` | Tensor Parallel plans / RNG / loss |
| `CP_*` | Context Parallel (sequence shard, attention merge) |
| `AMP_*` | Distributed AMP / grad clip / Inf flags |
| `DCP_*` | Distributed Checkpoint integrity & pickle |
| `PG_*` / `GLOO_*` | Process group / Gloo env divergence |
| `OPT_*` | Asymmetric optimizer steps under FSDP/TP |
| `hands_on_*` | Short “learn the API” baselines |
| `pickle_*` / `foreign_client.py` / `*_rpc*` | Store, RPC, serialization surfaces |

---

## Catalog by theme

### 1. Rendezvous, store, RPC, pickle

| Script | What it probes |
|--------|----------------|
| `group_demo.py` | Process groups / subgroups under `torchrun` |
| `foreign_client.py` | **Unauthenticated TCPStore**: connect as outsider, list/get/set/delete keys |
| `dsitributed_rpc.py` | Basic RPC init + `rpc_sync` |
| `pickle_udf_issue*.py`, `pickle_example_*.py` | RPC / pickle opcode surface (`_internal_rpc_pickler`, REDUCE/GLOBAL) |

**Takeaway:** Store and pickle paths are part of the security boundary; default rendezvous is not authenticated.

---

### 2. DDP — sync contracts, hooks, optimizer, AMP, checkpoints

Baselines and hook experiments:

| Script | Focus |
|--------|--------|
| `DDP_example_1.py` | Minimal DDP (CPU/gloo) |
| `DDP_example_malformed.py` | Malformed / broken DDP usage |
| `DDP_example_with_hook.py` … `_4.py` | Comm hooks behavior |
| `DDP_example_with_hook_leak_memory.py` | Hook memory leak |
| `DDP_example_with_hook_buf_exfer.py` | Buffer authority + Join (`R03`) |

Optimizer / RNG divergence (AllReduce grads ≠ synced **weights** or **Adam state**):

| Script | ID / idea |
|--------|-----------|
| `DDP_example_with_skip_optimizer.py` / `*_otim_not_synced.py` | **O01** — one rank skips `optimizer.step` |
| `DDP_example_with_different_lrs.py` | **O02** — different LR per rank |
| `DDP_example_with_unsynced_optimizer_state.py` | **O03** — Adam moments stay local |
| `DDP_example_with_rng_dropout_skew.py` | **O04** — per-rank RNG / dropout skew |

Hangs / AMP / checkpoints:

| Script | Focus |
|--------|--------|
| `DDP_example_static_graph_hang.py` | **H04** — lie about `static_graph` / unused params |
| `DDP_example_amp_nan_rank.py` | **N01** — Inf/NaN on one rank vs scaler + AllReduce |
| `DDP_example_amp_with hook.py` | **N03** — compress hook + AMP (`MODE=ok\|nan`) |
| `DDP_example_amp_mixed_precision_mismatch.py` | **N04** — dtype / stack mismatch on the wire |
| `DDP_example_checkpoint_pitfalls.py` | **K01–K06** — `module.` keys, skew save, stale ckpt, mid-step save |

---

### 3. DTensor — placements, mesh, meta, SPMD

Hands-on: `DTensor_basic.py`, `DTensor_run_check.py`.

Mismatch / poison probes (typically 2 terminals):

| Script | Focus |
|--------|--------|
| `DTensor_distribute_shard_mismatch_ranks.py` | Different `Shard` dims per rank |
| `DTensor_distribute_src_data_rank_poisoned.py` | `src_data_rank` / poisoned local data |
| `DTensor_distribute_uneven_shard.py` / `*_redistribute_uneven_shard.py` | Uneven shards |
| `DTensor_distribute_allow_implicit_replication.py` | Implicit replication |
| `DTensor_redistribute_shard_mismatch_ranks_*.py` | Redistribute placement skew |
| `DTensor_redistribute_partial_etc.py` | Partial / redistribute edge cases |
| `DTensor_mesh_mismatch_rank.py` | Mesh / dim-name mismatch |
| `DTensor_meta_mismatch_ranks.py` | `tensor_meta` mismatch |
| `DTensor_spmd_strategy_mismatch_ranks.py` | OpSchema / cost-model skew |
| `DTensor_spmd_strategy_hit.py` | Skew that **hits** on current PyTorch |
| `DTensor_spec_hash_cache.py` | Spec hash vs `__eq__` → sharding-prop cache confusion |
| `DTensor_fast_path_one_rank.py` | Fast path / cache restricted on one rank |
| `DTensor_split_factor_strided.py` | `split_factor` / `_StridedShard` on 2D mesh |

---

### 4. FSDP2 — unshard windows, sync, offload

Baseline: `hands_on_03_fsdp2.py`.

| Script | Focus |
|--------|--------|
| `FSDP_same_shape_swap.py` / `FSDP_pre_post_fw_bwd.py` / `FSDP_reshard_After_forward.py` | Module order / same-shape swap / reshard-after-forward |
| `FSDP_mixed_precision_diverge.py` | MixedPrecisionPolicy diverge |
| `FSDP_cpu_offload_pin_diverge.py` | CPU offload / `pin_memory` |
| `FSDP_ignored_params_diverge.py` | `ignored_params` |
| `FSDP_unshard_window_overwrite.py` | Poison during unshard window |
| `FSDP_skip_reduce_scatter.py` | Zero local grad contribution vs reduce-scatter |
| `FSDP_stale_unshard_memory.py` | Stale unshard buffers (`ref` / `realloc`) |
| `FSDP_unshard_order_diverge.py` | Explicit `unshard()` order |
| `FSDP_grad_accum_sync_diverge.py` | Grad accum / `reshard_after_backward` |
| `FSDP_data_triggered_desync.py` | Uneven steps / early exit |
| `FSDP_state_dict_unshard_vs_idle.py` | `state_dict` while idle vs during unshard |

---

### 5. Tensor Parallel (TP)

Baselines: `TP_basic_1.py`, `TP_basic_mutate_local+shard.py`, `hands_on`-style plan files.

| Script | Focus |
|--------|--------|
| `TP_init_divergent.py` | Skip / diverge TP init on one rank |
| `TP_plan_diverges.py` / `TP_plan_ip_op_mismatch.py` / `TP_plan_keys_diverge.py` | Plan / in–out / key mismatch |
| `TP_sequence_parallel_mismatch.py` | SequenceParallel on vs off (LayerNorm) |
| `TP_rng_dropout_mismatch.py` | RNG / dropout under TP |
| `TP_loss_allreduce_mismatch.py` | Skip / local / poison loss all-reduce |
| `TP_hook_diverge.py` | Different autograd hooks on TP DTensor params |

---

### 6. Context Parallel (CP) — attention merge & blast radius

Baseline: `hands_on_04_cp.py`.

| Script | Focus |
|--------|--------|
| `CP_rotate_method_diverge.py` / `CP_is_casual.py` / `CP_load_balance_diverge.py` | Rotate method / causal / load balance |
| `CP_omit_seq_buffer_diverge.py` / `CP_buffer_seq_dims_diverge.py` / `CP_no_restore_diverge.py` | Seq buffers / restore |
| `CP_causal_rope_offset_breakage.py` | Causal / RoPE position offset |
| `CP_packed_doc_mask_isolation.py` | Packed docs / mask isolation |
| `CP_poisoned_lse.py` | **Poisoned LSE** in online-softmax merge (`huge` / `neginf`) |
| `CP_poison_q_vs_kv_blast.py` | Blast radius: poison Q (local) vs K/V (ring-shared) |
| `CP_seq_not_divisible_world_size.py` | Seq len / packing not divisible |
| `CP_sdpa_backend_diverge.py` | SDPA backend pinned differently per rank |
| `CP_unshard_while_peer_in_context.py` | Unshard while peer still in CP context |

---

### 7. AMP / optimizer asymmetry

| Script | Focus |
|--------|--------|
| `AMP_found_inf_max_reduce.py` | `found_inf` **MAX** all-reduce → one Inf skips **all** ranks |
| `AMP_inf_weight_fsdp_reshard.py` | Inf in weights vs grads under FSDP reshard |
| `AMP_grad_norm_inflate.py` | Inflate local norm / skip clip all-reduce |
| `OPT_asym_step_fsdp_tp.py` | Double `step` / skip `zero_grad` under FSDP2 or TP |

---

### 8. Process groups & Gloo ops skew

| Script | Focus |
|--------|--------|
| `PG_backend_mix_mpi_gloo.py` | Mixing backends (unsupported; hangs under `torchrun`≠`mpirun`) |
| `GLOO_env_timeout_diverge.py` | Divergent timeouts / `GLOO_SOCKET_IFNAME` |

---

### 9. Distributed Checkpoint (DCP) — integrity & deserialization

| Script | Focus |
|--------|--------|
| `DCP_pickle_surface_audit.py` | Map pickle/`torch.load` sinks (S1–S6); harmless canary |
| `DCP_chunk_tamper_integrity.py` | Bitflip / swap FQN pointers / truncate — **no checksum** in stock DCP |
| `DCP_metadata_lie_reshard.py` | Lie about shapes/offsets/sizes then reshard load |
| `DCP_cross_topology_load.py` | Load across mesh factor / axis swap / shard dim (weak validation) |
| `DCP_async_mutate_inflight.py` | Mutate params while async save in flight |
| `DCP_async_purge_while_load.py` | `keep_latest_k`-style purge vs peer still loading |

**Takeaway:** Treat checkpoint directories like untrusted input unless you add integrity (e.g. external hashes/signatures). Contrast: some stacks layer `integrity.json` (SHA-256) *on top of* DCP.

---

## Suggested learning path

1. `DDP_example_1.py` → hooks → O01–O04 optimizer/RNG scripts  
2. `foreign_client.py` against a local `group_demo` / torchrun job (store trust)  
3. `DTensor_basic.py` → one shard-mismatch + `DTensor_spmd_strategy_hit.py`  
4. `hands_on_03_fsdp2.py` → unshard-window / reduce-scatter probes  
5. `TP_basic_1.py` → plan / loss all-reduce diverge  
6. `hands_on_04_cp.py` → `CP_poisoned_lse.py` / `CP_poison_q_vs_kv_blast.py`  
7. `DCP_pickle_surface_audit.py` → `DCP_chunk_tamper_integrity.py` → metadata / topology lies  

---

## Interpreting results

When you run a probe, record:

1. **Does it hang, throw, or finish?**  
2. If it finishes, are **weight / attention / checkpoint checksums** equal across ranks?  
3. Is the bad outcome **detectable** by stock PyTorch, or only by your own asserts?  
4. Would **network isolation**, **store ACL**, **weights_only / safe load**, or **external integrity** have stopped it?

Silent success with wrong numerics is usually the most important finding.

---

## Safety & ethics

- Use **localhost** and disposable ports (`2950`, `29500`, …).  
- Do not use these scripts to interfere with shared or production jobs.  
- Pickle / canary demos are for **mapping surfaces**, not for shipping payloads.  
- Findings are about **framework trust assumptions** and **operator footguns**; report responsibly if you escalate to upstream.

---

## Repo hygiene notes

- Duplicate / scratch copies exist (e.g. `* copy.py`); prefer the non-copy filename.  
- Some early DDP scripts still say `DDP_example_1.py` in the docstring `Run:` line — trust the **filename you invoke**.  
- Filenames keep historical typos (`distributed_rpc.py`) so existing run notes stay valid.

---

## License / status

Personal assessment / research notes and reproduction scripts. Not an official PyTorch package. APIs and default behaviors change across releases — re-verify on the torch version you care about.
