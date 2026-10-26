"""DCP chunk-tamper probe: is there any integrity check on load?

MODE=healthy|bitflip|swap_fqn|truncate

  Stock DCP (FileSystemWriter/Reader) stores:
    - ``*.distcp``  — tensor / BYTE_IO payloads (offset+length in .metadata)
    - ``.metadata`` — pickle map: FQN → storage path/offset/length (+ shapes)

  There is **no** checksum / hash / signature over chunks or metadata in
  ``torch.distributed.checkpoint``. Load trusts the store. Contrast: Megatron
  Core can write/verify ``integrity.json`` (SHA-256) *on top of* DCP.

  What each MODE does after a clean 2-key save:
    healthy   — load as-is; checksum of weights should match.
    bitflip   — flip one byte inside a ``.distcp`` payload region.
    swap_fqn  — in ``.metadata``, swap ``storage_data`` pointers for ``a`` vs ``b``.
                Debugs *why* this is silent:
                  (1) no checksum field on chunks / no hash verify on load
                  (2) load never checks that pointer values "belong" to that FQN —
                      it only follows path+offset+length (+ shape on copy)
                ``.distcp`` bytes are untouched (file hash unchanged).
    truncate  — shorten a ``.distcp`` below its recorded length.

  Outcome classes printed: ok | loud_error | silent_poison

Run (cuda_test / latest torch; single process):
  conda run -n cuda_test python -W ignore DCP_chunk_tamper_integrity.py
  MODE=bitflip conda run -n cuda_test python -W ignore DCP_chunk_tamper_integrity.py
  MODE=swap_fqn conda run -n cuda_test python -W ignore DCP_chunk_tamper_integrity.py
  MODE=truncate conda run -n cuda_test python -W ignore DCP_chunk_tamper_integrity.py
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader, _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _ptr_row(key: MetadataIndex, info: _StorageInfo) -> str:
    return (
        f"  fqn={key.fqn!r:3} → {info.relative_path}"
        f"  off={info.offset}  len={info.length}"
    )


def _save_clean(ckpt: Path) -> dict[str, torch.Tensor]:
    # Distinct, same-shape tensors so FQN swap can succeed silently.
    state = {
        "a": torch.arange(8, dtype=torch.float32),
        "b": torch.arange(8, dtype=torch.float32) + 100.0,
    }
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True)
    dcp.save(state, checkpoint_id=str(ckpt), no_dist=True)
    return {k: v.clone() for k, v in state.items()}


def _load(ckpt: Path) -> dict[str, torch.Tensor]:
    out = {
        "a": torch.zeros(8, dtype=torch.float32),
        "b": torch.zeros(8, dtype=torch.float32),
    }
    dcp.load(out, checkpoint_id=str(ckpt), no_dist=True)
    return out


def _distcp_files(ckpt: Path) -> list[Path]:
    return sorted(p for p in ckpt.iterdir() if p.suffix == ".distcp")


def _tamper_bitflip(ckpt: Path) -> None:
    files = _distcp_files(ckpt)
    if not files:
        raise RuntimeError("no .distcp files")
    p = files[0]
    data = bytearray(p.read_bytes())
    # Flip a mid-file byte; prefer payload over tiny headers when possible.
    i = max(len(data) // 2, min(64, len(data) - 1))
    data[i] ^= 0xFF
    p.write_bytes(bytes(data))
    _log(f"bitflip: {p.name}[{i}] ^= 0xFF  (len={len(data)})")


def _tamper_truncate(ckpt: Path) -> None:
    files = _distcp_files(ckpt)
    if not files:
        raise RuntimeError("no .distcp files")
    p = files[0]
    data = p.read_bytes()
    new_len = max(len(data) // 2, 8)
    p.write_bytes(data[:new_len])
    _log(f"truncate: {p.name} {len(data)} → {new_len}")


def _tamper_swap_fqn(ckpt: Path) -> None:
    """Swap storage pointers for MetadataIndex(fqn='a') <-> 'b'.

    Separates two failure modes the user cares about:
      A) checksum — ``_StorageInfo`` has no digest; load never hashes bytes.
         Also: ``.distcp`` content is unchanged, so a *file* checksum would
         still match after this tamper.
      B) pointer binding — load never checks that the pointer values are the
         ones originally recorded for that FQN; it just dereferences whatever
         ``storage_data[MetadataIndex(fqn=...)]`` currently says.
    """
    meta_path = ckpt / ".metadata"
    distcps = _distcp_files(ckpt)
    file_hash_before = {p.name: _sha256(p) for p in distcps}

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # _StorageInfo fields: prove there is nothing to checksum against.
    info0 = next(iter(meta.storage_data.values()))
    _log(
        f"swap_fqn debug A (checksum): _StorageInfo fields = "
        f"{sorted(info0.__dict__.keys())}  "
        f"(no checksum/digest/sha — nothing for load to verify)"
    )

    by_fqn: dict[str, list] = {}
    for key in list(meta.storage_data.keys()):
        by_fqn.setdefault(key.fqn, []).append(key)

    if "a" not in by_fqn or "b" not in by_fqn:
        raise RuntimeError(f"expected fqn a/b in storage_data, got {list(by_fqn)}")

    ka, kb = by_fqn["a"][0], by_fqn["b"][0]
    _log("swap_fqn debug B (pointers) BEFORE:")
    _log(_ptr_row(ka, meta.storage_data[ka]))
    _log(_ptr_row(kb, meta.storage_data[kb]))

    # Only rewrite .metadata map entries — leave .distcp bytes alone.
    meta.storage_data[ka], meta.storage_data[kb] = (
        meta.storage_data[kb],
        meta.storage_data[ka],
    )
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    _log("swap_fqn debug B (pointers) AFTER (values crossed; still structurally valid):")
    _log(_ptr_row(ka, meta.storage_data[ka]))
    _log(_ptr_row(kb, meta.storage_data[kb]))

    file_hash_after = {p.name: _sha256(p) for p in distcps}
    _log(
        f"swap_fqn debug A cont: .distcp sha256[:16] before={file_hash_before} "
        f"after={file_hash_after}  unchanged={file_hash_before == file_hash_after}"
    )
    _log(
        "swap_fqn conclusion: silent if load trusts pointer table as-is. "
        "A file checksum would NOT catch this; you'd need "
        "FQN↔content binding (or signed metadata) — which DCP does not check."
    )


def _classify(orig: dict[str, torch.Tensor], loaded: dict[str, torch.Tensor]) -> str:
    if torch.equal(loaded["a"], orig["a"]) and torch.equal(loaded["b"], orig["b"]):
        return "ok"
    # Swapped a<->b is the silent poison we look for.
    if torch.equal(loaded["a"], orig["b"]) and torch.equal(loaded["b"], orig["a"]):
        return "silent_poison (a/b swapped)"
    if not torch.equal(loaded["a"], orig["a"]) or not torch.equal(loaded["b"], orig["b"]):
        return "silent_poison (values differ, not clean swap)"
    return "ok"


def _has_builtin_integrity() -> None:
    # Document what's in this torch build.
    reader_src = Path(FileSystemReader.__module__.replace(".", "/"))
    # Quick attribute / docstring sniff
    fs_mod = __import__(
        "torch.distributed.checkpoint.filesystem", fromlist=["*"]
    )
    text = Path(fs_mod.__file__).read_text(encoding="utf-8")
    hits = [
        w
        for w in ("checksum", "sha256", "sha1", "blake", "digest", "integrity", "crc32")
        if w in text.lower()
    ]
    _log(
        f"integrity scan filesystem.py: "
        f"{'no hash/checksum tokens' if not hits else hits}"
    )
    _log(f"MetadataIndex fields (binding is path+offset+len only): {MetadataIndex.__annotations__ if hasattr(MetadataIndex, '__annotations__') else MetadataIndex}")


def main() -> None:
    mode = os.environ.get("MODE", "healthy")
    _log(f"torch={torch.__version__}  MODE={mode}")
    _has_builtin_integrity()

    ckpt = Path(tempfile.mkdtemp(prefix="dcp_tamper_"))
    try:
        orig = _save_clean(ckpt)
        _log(f"saved: a={orig['a'].tolist()}  b={orig['b'].tolist()}")
        _log(f"files: {[p.name for p in sorted(ckpt.iterdir())]}")

        if mode == "bitflip":
            _tamper_bitflip(ckpt)
        elif mode == "swap_fqn":
            _tamper_swap_fqn(ckpt)
        elif mode == "truncate":
            _tamper_truncate(ckpt)
        elif mode == "healthy":
            pass
        else:
            raise SystemExit(f"unknown MODE={mode}")

        try:
            loaded = _load(ckpt)
        except BaseException as e:
            # CheckpointException subclasses BaseException, not Exception.
            msg = str(e).splitlines()[0][:160]
            _log(f"outcome: loud_error  {type(e).__name__}: {msg}")
            return

        outcome = _classify(orig, loaded)
        _log(f"loaded: a={loaded['a'].tolist()}  b={loaded['b'].tolist()}")
        _log(f"outcome: {outcome}")
    finally:
        shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    main()
