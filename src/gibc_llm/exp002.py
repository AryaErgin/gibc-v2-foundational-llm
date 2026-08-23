"""EXP-002 frozen-artifact and prefix-control helpers."""

from pathlib import Path

from .utils import sha256_file


def verify_exp001_prefix(stream_path: Path, prefix_bytes: int, expected_sha256: str) -> bool:
    """Verify the exact EXP-001 uint16 byte prefix without loading it into RAM."""
    import hashlib

    digest = hashlib.sha256()
    with Path(stream_path).open("rb") as handle:
        remaining = prefix_bytes
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                return False
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest() == expected_sha256


def assert_frozen_exp001_artifacts(tokenizer_path: Path, validation_path: Path) -> None:
    if sha256_file(tokenizer_path) != "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14":
        raise RuntimeError("EXP-002 requires the exact frozen EXP-001 tokenizer artifact.")
    import torch
    from .data import tensor_sha256
    values = torch.load(validation_path, map_location="cpu", weights_only=True)
    if tensor_sha256(values["inputs"]) != "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7" or tensor_sha256(values["targets"]) != "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e":
        raise RuntimeError("EXP-002 requires the exact frozen EXP-001 validation artifact.")
