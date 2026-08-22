"""From-scratch byte-level BPE training for EXP-001."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .data import Document
from .utils import sha256_file


@dataclass
class TokenizerArtifact:
    tokenizer: Tokenizer
    path: Path
    sha256: str
    vocab_size: int
    special_tokens: list[str]


def train_tokenizer(
    documents: Iterable[Document], output_dir: Path, vocab_size: int, eod_token: str
) -> TokenizerArtifact:
    """Train a byte-level BPE solely on already-selected training documents."""
    texts: list[str] = []
    for document in documents:
        if document.split != "train":
            raise ValueError("Tokenizer training accepts training documents only.")
        texts.append(document.text)
    if not texts:
        raise ValueError("Tokenizer training requires at least one training document.")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        texts,
        trainer=BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[eod_token],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != vocab_size:
        raise RuntimeError(f"Byte-level BPE produced {actual_vocab_size} entries, expected exactly {vocab_size}.")
    if tokenizer.token_to_id(eod_token) is None:
        raise RuntimeError("EXP-001 EOD token was not serialized into the tokenizer vocabulary.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tokenizer.json"
    tokenizer.save(str(path))
    return TokenizerArtifact(
        tokenizer=tokenizer,
        path=path,
        sha256=sha256_file(path),
        vocab_size=actual_vocab_size,
        special_tokens=[eod_token],
    )


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
