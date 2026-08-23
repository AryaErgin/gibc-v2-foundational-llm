import hashlib
from pathlib import Path

from gibc_llm.exp002 import verify_exp001_prefix


def test_exp002_prefix_verifier_accepts_only_the_exact_exp001_byte_prefix(tmp_path: Path) -> None:
    """Breaks if EXP-002 can proceed without byte-identical first-100M stream control."""
    prefix = bytes(range(64)) * 8
    stream = tmp_path / "exp002.uint16"
    stream.write_bytes(prefix + b"new deterministic continuation")
    assert verify_exp001_prefix(stream, len(prefix), hashlib.sha256(prefix).hexdigest())
    assert not verify_exp001_prefix(stream, len(prefix), "0" * 64)
