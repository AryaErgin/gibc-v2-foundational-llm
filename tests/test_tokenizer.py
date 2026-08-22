from pathlib import Path
import random
import string

import pytest

from gibc_llm.data import Document
from gibc_llm.tokenizer import train_tokenizer


def _rich_training_documents() -> list[Document]:
    rng = random.Random(123)
    alphabet = string.ascii_letters + string.digits
    text = " ".join("".join(rng.choices(alphabet, k=24)) for _ in range(30_000))
    return [Document(document_id="train-fixture", text=text, split="train")]


def test_byte_level_bpe_has_exact_total_vocabulary_and_eod_only(tmp_path: Path) -> None:
    """Breaks if tokenizer training changes the fixed 8,192 total vocabulary or special-token set."""
    artifact = train_tokenizer(_rich_training_documents(), output_dir=tmp_path, vocab_size=8192, eod_token="<|endoftext|>")

    assert artifact.vocab_size == 8192
    assert artifact.special_tokens == ["<|endoftext|>"]
    assert artifact.tokenizer.token_to_id("<|endoftext|>") is not None
    assert artifact.tokenizer.token_to_id("<unk>") is None
    assert len(artifact.sha256) == 64


def test_tokenizer_rejects_non_training_documents(tmp_path: Path) -> None:
    """Breaks if held-out validation text can enter tokenizer training."""
    documents = [Document(document_id="validation-fixture", text="must not train", split="validation")]

    with pytest.raises(ValueError, match="training documents"):
        train_tokenizer(documents, output_dir=tmp_path, vocab_size=8192, eod_token="<|endoftext|>")
