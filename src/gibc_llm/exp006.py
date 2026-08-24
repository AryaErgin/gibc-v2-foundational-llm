"""EXP-006 preparation: deterministic 3x Data Recipe v1 replay with a hard EXP-004 prefix gate."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import build_benchmark_filter, write_token_stream
from .exp004 import (
    SOURCE_ORDER,
    GlobalDeduplicatedTokenMixer,
    _screened_train_documents,
    _validation_record,
    assert_frozen_exp004_artifacts,
)
from .exp003 import FROZEN_TOKENIZER_SHA256
from .utils import atomic_json_write, sha256_file, sha256_file_prefix


EXP004_PREFIX_SHA256 = "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1"
EXP004_PREFIX_STORED_TOKEN_IDS = 300_023_809
EXP004_PREFIX_BYTE_COUNT = EXP004_PREFIX_STORED_TOKEN_IDS * 2


def verify_stream_prefix(stream_path: Path, byte_count: int, expected_sha256: str) -> str:
    """Hash an exact raw prefix and reject any mismatch before a manifest can authorize training."""
    observed = sha256_file_prefix(stream_path, byte_count)
    if observed != expected_sha256:
        raise RuntimeError(
            f"EXP-006 prefix SHA-256 mismatch: expected {expected_sha256}, observed {observed}. "
            "The artifact is not authorized for training."
        )
    return observed


def assert_frozen_exp006_source(exp004_artifact_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return only validated EXP-004 Recipe v1 controls; never regenerate them for EXP-006."""
    source = Path(exp004_artifact_dir)
    tokenizer = source / "tokenizer" / "tokenizer.json"
    general_validation = source / "general_validation.pt"
    edu_validation = source / "edu_validation.pt"
    stream = source / "train-token-stream.uint16"
    assert_frozen_exp004_artifacts(tokenizer, general_validation, edu_validation)
    if not stream.is_file() or stream.stat().st_size != EXP004_PREFIX_BYTE_COUNT:
        raise RuntimeError("EXP-006 requires the complete exact EXP-004 uint16 stream as its frozen prefix source.")
    if sha256_file(stream) != EXP004_PREFIX_SHA256:
        raise RuntimeError("EXP-006 requires the exact frozen EXP-004 stream SHA-256.")
    return tokenizer, general_validation, edu_validation, stream


def prepare_exp006(config: Any, artifact_dir: Path, exp004_artifact_dir: Path) -> dict[str, Any]:
    """Materialize a 3x non-cycled Recipe v1 stream, then hard-verify its EXP-004 raw-byte prefix; never trains."""
    from .tokenizer import load_tokenizer

    if config.experiment_id != "EXP-006" or config.mixture is None:
        raise ValueError("prepare_exp006 requires the approved configs/exp006.yaml mixture specification.")
    expected_targets = {"fineweb": 600_047_616, "fineweb_edu": 300_023_808}
    if config.mixture.get("target_prediction_tokens") != expected_targets:
        raise RuntimeError("EXP-006 must retain the exact 2:1 600M/300M prediction-token targets.")

    started = time.perf_counter()
    frozen_tokenizer, frozen_general, frozen_edu, frozen_stream = assert_frozen_exp006_source(exp004_artifact_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "tokenizer").mkdir(exist_ok=True)
    tokenizer_path = artifact_dir / "tokenizer" / "tokenizer.json"
    general_path = artifact_dir / "general_validation.pt"
    edu_path = artifact_dir / "edu_validation.pt"
    shutil.copy2(frozen_tokenizer, tokenizer_path)
    shutil.copy2(frozen_general, general_path)
    shutil.copy2(frozen_edu, edu_path)
    if sha256_file(tokenizer_path) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-006 copied tokenizer differs from the frozen EXP tokenizer hash.")

    tokenizer = load_tokenizer(tokenizer_path)
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("Frozen tokenizer does not contain the required EOD token.")
    cache_dir = artifact_dir / "cache"
    contamination_filter, benchmark_sources = build_benchmark_filter(cache_dir / "benchmarks", config.data.contamination_ngram_size)
    source_configs = {
        "fineweb": config.data,
        "fineweb_edu": replace(
            config.data,
            dataset_repo="HuggingFaceFW/fineweb-edu",
            dataset_config="default",
            dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        ),
    }
    source_counters = {
        source: {"scanned_documents": 0, "accepted_documents": 0, "rejected_documents": 0, "validation_documents_excluded": 0}
        for source in SOURCE_ORDER
    }
    documents = {
        source: _screened_train_documents(source_configs[source], cache_dir / source, contamination_filter, source_counters[source])
        for source in SOURCE_ORDER
    }
    stored_tokens = config.training.full_training_tokens + 1
    mixer = GlobalDeduplicatedTokenMixer(documents, tokenizer, eod_id, expected_targets, stored_tokens)
    stream_path = artifact_dir / "train-token-stream.uint16"
    stream = write_token_stream(stream_path, mixer, stored_tokens, config.data.context_length)
    if sum(mixer.prediction_token_contributions.values()) != config.training.full_training_tokens:
        raise RuntimeError("EXP-006 source contributions do not account for every prediction token exactly once.")
    observed_prefix_sha256 = verify_stream_prefix(stream_path, EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256)

    general_values = torch.load(general_path, map_location="cpu", weights_only=True)
    edu_values = torch.load(edu_path, map_location="cpu", weights_only=True)
    manifest = {
        "experiment_id": "EXP-006",
        "preparation_mode": "full_stream",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "mixture": {
            **config.mixture,
            "deterministic_mixing_method": "token-deficit-balanced whole-document selection; FineWeb wins deterministic ties",
            "actual_prediction_token_contributions": mixer.prediction_token_contributions,
            "actual_stored_token_contributions": mixer.stored_token_contributions,
            "documents_contributed": mixer.documents_contributed,
            "cross_source_duplicates_skipped": mixer.cross_source_duplicates_skipped,
            "intra_source_duplicates_skipped": mixer.intra_source_duplicates_skipped,
            "unique_document_count": len(mixer.selected_document_ids),
        },
        "split": {"method": "sha256(seed:canonical_content_sha256) modulo buckets", "seed": config.data.split_seed, "validation_buckets": config.data.validation_bucket_cutoff, "modulus": config.data.validation_bucket_modulus},
        "contamination": {
            "method": "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap",
            "ngram_size": config.data.contamination_ngram_size,
            "benchmark_sources": benchmark_sources,
            "sources": source_counters,
        },
        "frozen_exp004_source": {"stream_sha256": sha256_file(frozen_stream), "stored_token_ids": EXP004_PREFIX_STORED_TOKEN_IDS},
        "exp004_prefix": {
            "byte_count": EXP004_PREFIX_BYTE_COUNT,
            "expected_sha256": EXP004_PREFIX_SHA256,
            "observed_sha256": observed_prefix_sha256,
            "prefix_match": True,
        },
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": sha256_file(tokenizer_path), "vocab_size": 8192, "special_tokens": [config.data.eod_token]},
        "general_validation": _validation_record(general_path, general_values),
        "edu_validation": {**_validation_record(edu_path, edu_values), "contamination_screened": True, "frozen_from": str(frozen_edu)},
        "packed": {
            "representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views",
            "storage_dtype": "uint16",
            "context_length": config.data.context_length,
            "prediction_tokens_per_example": config.training.sequence_predictions,
            "train_prediction_tokens": config.training.full_training_tokens,
            "train_token_count_including_final_target": stored_tokens,
            "train_examples": len(stream),
            "train_stream_file": stream_path.name,
            "train_stream_bytes": stream_path.stat().st_size,
            "train_stream_sha256": sha256_file(stream_path),
            "non_cycled": True,
        },
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    atomic_json_write(artifact_dir / "manifest.json", manifest)
    return manifest
