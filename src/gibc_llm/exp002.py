"""EXP-002 frozen-artifact and prefix-control helpers."""

import itertools
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

from .utils import sha256_file

EXP001_PREFIX_BYTES = 200_015_874
EXP001_PREFIX_SHA256 = "86b84dc30f88ac1ba8daee4f7b160f581d3e9a5987fbf86fff5dbab967647d04"


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


def prepare_exp002(config: Any, artifact_dir: Path, exp001_artifact_dir: Path) -> dict[str, Any]:
    """Reconstruct one deterministic 300M stream with frozen EXP-001 artifacts."""
    import torch
    from .data import Document, assign_split, build_benchmark_filter, iter_fineweb_documents, resolve_fineweb_revision, stable_document_id, tensor_sha256, write_token_stream
    from .tokenizer import load_tokenizer
    from .utils import atomic_json_write

    if config.experiment_id != "EXP-002":
        raise ValueError("prepare_exp002 requires configs/exp002.yaml.")
    started = time.perf_counter()
    source = Path(exp001_artifact_dir)
    frozen_tokenizer = source / "tokenizer" / "tokenizer.json"
    frozen_validation = source / "validation.pt"
    assert_frozen_exp001_artifacts(frozen_tokenizer, frozen_validation)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "tokenizer").mkdir(exist_ok=True)
    shutil.copy2(frozen_tokenizer, artifact_dir / "tokenizer" / "tokenizer.json")
    shutil.copy2(frozen_validation, artifact_dir / "validation.pt")
    tokenizer = load_tokenizer(frozen_tokenizer)
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("Frozen EXP-001 tokenizer has no EOD token.")
    cache_dir = artifact_dir / "cache"
    revision = resolve_fineweb_revision(config.data)
    contamination_filter, benchmark_sources = build_benchmark_filter(cache_dir / "benchmarks", config.data.contamination_ngram_size)
    initial_train: list[Document] = []
    scanned = accepted = rejected = train_bytes = 0
    validation_bytes = 0
    validation_floor = max(2 * 1024 * 1024, config.training.smoke_validation_tokens * 8)
    source_documents = iter(iter_fineweb_documents(config.data, revision, cache_dir / "fineweb"))
    for text in source_documents:
        scanned += 1
        document_id = stable_document_id(text)
        split = assign_split(document_id, config.data.split_seed, config.data.validation_bucket_modulus, config.data.validation_bucket_cutoff)
        if contamination_filter.screen(text).rejected:
            rejected += 1
            continue
        accepted += 1
        if split == "train":
            initial_train.append(Document(document_id, text, split))
            train_bytes += len(text.encode("utf-8"))
        else:
            validation_bytes += len(text.encode("utf-8"))
        if train_bytes >= config.data.tokenizer_training_text_bytes and validation_bytes >= validation_floor:
            break
    documents_contributed = 0
    def tokens() -> Iterator[int]:
        nonlocal scanned, accepted, rejected, documents_contributed
        for document in initial_train:
            documents_contributed += 1
            yield from tokenizer.encode(document.text).ids
            yield eod_id
        for text in source_documents:
            scanned += 1
            document_id = stable_document_id(text)
            split = assign_split(document_id, config.data.split_seed, config.data.validation_bucket_modulus, config.data.validation_bucket_cutoff)
            if contamination_filter.screen(text).rejected:
                rejected += 1
                continue
            accepted += 1
            if split == "train":
                documents_contributed += 1
                yield from tokenizer.encode(text).ids
                yield eod_id
    stored = config.training.full_training_tokens + 1
    stream_path = artifact_dir / "train-token-stream.uint16"
    stream = write_token_stream(stream_path, itertools.islice(tokens(), stored), stored, config.data.context_length)
    observed = _prefix_hash(stream_path, EXP001_PREFIX_BYTES)
    if observed != EXP001_PREFIX_SHA256:
        raise RuntimeError("EXP-002 deterministic reconstruction prefix mismatch; training is invalid.")
    validation = torch.load(artifact_dir / "validation.pt", map_location="cpu", weights_only=True)
    manifest = {"experiment_id": "EXP-002", "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": revision, "field": config.data.text_field}, "split": {"method": "sha256(seed:canonical_content_sha256) modulo buckets", "seed": config.data.split_seed, "validation_buckets": config.data.validation_bucket_cutoff, "modulus": config.data.validation_bucket_modulus}, "contamination": {"method": "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap", "ngram_size": config.data.contamination_ngram_size, "benchmark_sources": benchmark_sources, "scanned_documents": scanned, "accepted_documents": accepted, "rejected_documents": rejected}, "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": sha256_file(frozen_tokenizer), "vocab_size": 8192, "special_tokens": [config.data.eod_token]}, "validation": {"file": "validation.pt", "prediction_tokens": int(validation["targets"].numel()), "inputs_sha256": tensor_sha256(validation["inputs"]), "targets_sha256": tensor_sha256(validation["targets"])}, "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": config.training.full_training_tokens, "train_token_count_including_final_target": stored, "train_examples": len(stream), "train_stream_file": stream_path.name, "train_stream_bytes": stream_path.stat().st_size, "train_stream_sha256": sha256_file(stream_path), "train_documents_contributed": documents_contributed, "non_cycled": True}, "exp001_prefix": {"byte_count": EXP001_PREFIX_BYTES, "expected_sha256": EXP001_PREFIX_SHA256, "observed_sha256": observed, "prefix_match": True}, "preparation_wall_seconds": time.perf_counter() - started}
    atomic_json_write(artifact_dir / "manifest.json", manifest)
    return manifest


def _prefix_hash(stream_path: Path, prefix_bytes: int) -> str:
    import hashlib
    digest = hashlib.sha256()
    with stream_path.open("rb") as handle:
        for block in iter(lambda: handle.read(min(1024 * 1024, prefix_bytes)), b""):
            digest.update(block)
            prefix_bytes -= len(block)
            if prefix_bytes == 0:
                break
    return digest.hexdigest() if prefix_bytes == 0 else ""
