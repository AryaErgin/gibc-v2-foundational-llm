"""EXP-011 preparation: deterministic 1.5B Data Recipe v1 extension with a hard EXP-006 prefix gate."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import build_benchmark_filter, write_token_stream
from .exp003 import FROZEN_TOKENIZER_SHA256
from .exp004 import (
    SOURCE_ORDER,
    GlobalDeduplicatedTokenMixer,
    _screened_train_documents,
    _validation_record,
    assert_frozen_exp004_artifacts,
)
from .exp006 import EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256, EXP004_PREFIX_STORED_TOKEN_IDS, verify_stream_prefix
from .utils import atomic_json_write, sha256_file, sha256_file_prefix


EXP006_PREDICTION_TOKENS = 900_071_424
EXP006_PREFIX_STORED_TOKEN_IDS = EXP006_PREDICTION_TOKENS + 1
EXP006_PREFIX_BYTE_COUNT = EXP006_PREFIX_STORED_TOKEN_IDS * 2


def verify_exp006_prefix(stream_path: Path, byte_count: int, expected_sha256: str) -> str:
    """Hash the exact 900M raw prefix and hard-fail before authorizing a 1.5B artifact."""
    observed = sha256_file_prefix(stream_path, byte_count)
    if observed != expected_sha256:
        raise RuntimeError(
            f"EXP-011 EXP-006 prefix SHA-256 mismatch: expected {expected_sha256}, observed {observed}. "
            "The artifact is not authorized for resume."
        )
    return observed


def assert_frozen_exp006_source(exp006_artifact_dir: Path) -> tuple[Path, Path, Path, Path, dict[str, Any], str]:
    """Return only the independently validated immutable EXP-006 900M controls."""
    source = Path(exp006_artifact_dir)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("EXP-011 requires the existing EXP-006 manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = manifest.get("packed", {})
    stream = source / packed.get("train_stream_file", "train-token-stream.uint16")
    if (
        manifest.get("experiment_id") != "EXP-006"
        or packed.get("train_prediction_tokens") != EXP006_PREDICTION_TOKENS
        or packed.get("train_token_count_including_final_target") != EXP006_PREFIX_STORED_TOKEN_IDS
        or packed.get("train_stream_bytes") != EXP006_PREFIX_BYTE_COUNT
        or not packed.get("train_stream_sha256")
        or not stream.is_file()
        or stream.stat().st_size != EXP006_PREFIX_BYTE_COUNT
        or sha256_file(stream) != packed["train_stream_sha256"]
    ):
        raise RuntimeError("EXP-011 requires the exact immutable full EXP-006 900M stream and manifest hash.")
    prefix = manifest.get("exp004_prefix", {})
    if (
        prefix.get("byte_count") != EXP004_PREFIX_BYTE_COUNT
        or prefix.get("expected_sha256") != EXP004_PREFIX_SHA256
        or prefix.get("observed_sha256") != EXP004_PREFIX_SHA256
        or prefix.get("prefix_match") is not True
        or verify_stream_prefix(stream, EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256) != EXP004_PREFIX_SHA256
    ):
        raise RuntimeError("EXP-011 requires EXP-006's independently verified exact EXP-004 prefix.")
    tokenizer = source / "tokenizer" / "tokenizer.json"
    general = source / "general_validation.pt"
    edu = source / "edu_validation.pt"
    assert_frozen_exp004_artifacts(tokenizer, general, edu)
    if sha256_file(tokenizer) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-011 source tokenizer is not the frozen 8192-entry tokenizer.")
    return tokenizer, general, edu, stream, manifest, sha256_file(manifest_path)


def prepare_exp011(config: Any, artifact_dir: Path, exp006_artifact_dir: Path) -> dict[str, Any]:
    """Build a full non-cycled 1.5B stream and prove its exact EXP-006/EXP-004 byte prefixes; never trains."""
    from .tokenizer import load_tokenizer

    expected_targets = {"fineweb": 1_000_013_824, "fineweb_edu": 500_006_912}
    if config.experiment_id != "EXP-011" or config.mixture is None or config.mixture.get("target_prediction_tokens") != expected_targets:
        raise ValueError("prepare_exp011 requires the approved configs/exp011.yaml 1.5B 2:1 mixture specification.")
    started = time.perf_counter()
    frozen_tokenizer, frozen_general, frozen_edu, frozen_stream, frozen_manifest, frozen_manifest_sha = assert_frozen_exp006_source(exp006_artifact_dir)
    frozen_stream_sha = sha256_file(frozen_stream)

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
        raise RuntimeError("EXP-011 copied tokenizer differs from the frozen tokenizer hash.")

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
        raise RuntimeError("EXP-011 source contributions do not account for every prediction token exactly once.")
    observed_exp006_prefix = verify_exp006_prefix(stream_path, EXP006_PREFIX_BYTE_COUNT, frozen_stream_sha)
    observed_exp004_prefix = verify_stream_prefix(stream_path, EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256)

    general_values = torch.load(general_path, map_location="cpu", weights_only=True)
    edu_values = torch.load(edu_path, map_location="cpu", weights_only=True)
    manifest = {
        "experiment_id": "EXP-011",
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
        "frozen_exp006_source": {
            "stream_sha256": frozen_stream_sha,
            "manifest_sha256": frozen_manifest_sha,
            "stored_token_ids": EXP006_PREFIX_STORED_TOKEN_IDS,
            "prediction_tokens": EXP006_PREDICTION_TOKENS,
            "source_manifest_experiment_id": frozen_manifest["experiment_id"],
        },
        "exp006_prefix": {
            "byte_count": EXP006_PREFIX_BYTE_COUNT,
            "expected_sha256": frozen_stream_sha,
            "observed_sha256": observed_exp006_prefix,
            "prefix_match": True,
        },
        "exp004_prefix": {
            "byte_count": EXP004_PREFIX_BYTE_COUNT,
            "expected_sha256": EXP004_PREFIX_SHA256,
            "observed_sha256": observed_exp004_prefix,
            "prefix_match": True,
            "stored_token_ids": EXP004_PREFIX_STORED_TOKEN_IDS,
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
