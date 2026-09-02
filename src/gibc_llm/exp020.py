"""EXP-020 deterministic 7.2B rebuild from stream zero with immutable prefix gates."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import NgramContaminationFilter, write_token_stream
from .exp003 import FROZEN_TOKENIZER_SHA256
from .exp004 import SOURCE_ORDER, GlobalDeduplicatedTokenMixer, _screened_train_documents, _validation_record
from .exp012 import (
    EXP011_PREFIX_BYTE_COUNT,
    EXP011_STREAM_SHA256,
    EXP012_TARGETS,
    assert_exp012_prefix_provenance,
    copy_exp011_benchmark_index,
)
from .full_run import EXP012_PREDICTION_TOKENS, EXP012_STREAM_SHA256
from .utils import atomic_json_write, sha256_file, sha256_file_prefix

EXP020_PREDICTION_TOKENS = 7_199_981_568
EXP020_STORED_TOKEN_IDS = EXP020_PREDICTION_TOKENS + 1
EXP020_TARGETS = {"fineweb": 4_799_987_712, "fineweb_edu": 2_399_993_856}
EXP012_STORED_TOKEN_IDS = EXP012_PREDICTION_TOKENS + 1
EXP012_PREFIX_BYTE_COUNT = EXP012_STORED_TOKEN_IDS * 2
EXP012_MANIFEST_SHA256 = "b19b508dd1d1928b8e3bbdf586547791dc3bd76af19f6e55b8c39465bd749ccf"


def assert_frozen_exp012_source(exp012_artifact_dir: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """Return the exact 2.4B source artifacts that seed an EXP-020 rebuild."""
    source = Path(exp012_artifact_dir)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != EXP012_MANIFEST_SHA256:
        raise RuntimeError("EXP-020 requires the audited EXP-012 manifest SHA-256.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = manifest.get("packed", {})
    stream = source / packed.get("train_stream_file", "train-token-stream.uint16")
    tokenizer = source / "tokenizer" / "tokenizer.json"
    general, edu = source / "general_validation.pt", source / "edu_validation.pt"
    if (
        manifest.get("experiment_id") != "EXP-012"
        or packed.get("train_prediction_tokens") != EXP012_PREDICTION_TOKENS
        or packed.get("train_token_count_including_final_target") != EXP012_STORED_TOKEN_IDS
        or packed.get("train_stream_bytes") != EXP012_PREFIX_BYTE_COUNT
        or packed.get("train_stream_sha256") != EXP012_STREAM_SHA256
        or not stream.is_file()
        or stream.stat().st_size != EXP012_PREFIX_BYTE_COUNT
        or sha256_file(stream) != EXP012_STREAM_SHA256
        or not tokenizer.is_file()
        or sha256_file(tokenizer) != FROZEN_TOKENIZER_SHA256
        or not general.is_file()
        or not edu.is_file()
    ):
        raise RuntimeError("EXP-020 requires the exact verified EXP-012 source artifacts.")
    assert_exp012_prefix_provenance(manifest, stream)
    return tokenizer, general, edu, stream, manifest


def assert_exp020_prefix_provenance(manifest: dict[str, Any], stream_path: Path) -> None:
    """Require an independently hashed EXP-012 prefix and inherited EXP-011 chain."""
    frozen = manifest.get("frozen_exp012_source", {})
    prefix = manifest.get("exp012_prefix", {})
    if (
        frozen.get("manifest_sha256") != EXP012_MANIFEST_SHA256
        or frozen.get("stream_sha256") != EXP012_STREAM_SHA256
        or frozen.get("stored_token_ids") != EXP012_STORED_TOKEN_IDS
        or frozen.get("prediction_tokens") != EXP012_PREDICTION_TOKENS
        or prefix.get("byte_count") != EXP012_PREFIX_BYTE_COUNT
        or prefix.get("expected_sha256") != EXP012_STREAM_SHA256
        or prefix.get("observed_sha256") != EXP012_STREAM_SHA256
        or prefix.get("prefix_match") is not True
        or sha256_file_prefix(stream_path, EXP012_PREFIX_BYTE_COUNT) != EXP012_STREAM_SHA256
    ):
        raise RuntimeError("EXP-020 manifest lacks a verified exact EXP-012 prefix chain.")
    inherited = manifest.get("exp011_prefix", {})
    if (
        inherited.get("expected_sha256") != EXP011_STREAM_SHA256
        or inherited.get("observed_sha256") != EXP011_STREAM_SHA256
        or inherited.get("prefix_match") is not True
        or inherited.get("byte_count") != EXP011_PREFIX_BYTE_COUNT
        or sha256_file_prefix(stream_path, EXP011_PREFIX_BYTE_COUNT) != EXP011_STREAM_SHA256
    ):
        raise RuntimeError("EXP-020 manifest lacks the inherited verified EXP-011 prefix chain.")


def prepare_exp020(config: Any, artifact_dir: Path, exp012_artifact_dir: Path, recorded_source_commit: str) -> dict[str, Any]:
    """Rebuild a unique 7.2B stream from zero; never append an insufficiently stateful artifact."""
    from .tokenizer import load_tokenizer

    if len(recorded_source_commit) != 40 or any(character not in "0123456789abcdef" for character in recorded_source_commit):
        raise ValueError("prepare_exp020 requires the immutable 40-character builder source commit.")
    if (
        config.experiment_id != "EXP-020"
        or config.mixture is None
        or config.mixture.get("target_prediction_tokens") != EXP020_TARGETS
        or config.training.full_training_tokens != EXP020_PREDICTION_TOKENS
    ):
        raise ValueError("prepare_exp020 requires the frozen 7.2B 2:1 EXP-020 configuration.")
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        raise RuntimeError("EXP-020 artifact directory already exists; preserve it and choose a new directory.")
    started = time.perf_counter()
    frozen_tokenizer, frozen_general, frozen_edu, frozen_stream, frozen_manifest = assert_frozen_exp012_source(exp012_artifact_dir)
    contamination = frozen_manifest.get("contamination", {})
    if (
        contamination.get("method") != "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap"
        or contamination.get("ngram_size") != config.data.contamination_ngram_size
        or not isinstance(contamination.get("benchmark_sources"), list)
    ):
        raise RuntimeError("EXP-020 requires exact recorded EXP-012 contamination provenance.")

    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "tokenizer").mkdir()
    tokenizer_path = artifact_dir / "tokenizer" / "tokenizer.json"
    general_path, edu_path = artifact_dir / "general_validation.pt", artifact_dir / "edu_validation.pt"
    shutil.copy2(frozen_tokenizer, tokenizer_path)
    shutil.copy2(frozen_general, general_path)
    shutil.copy2(frozen_edu, edu_path)
    tokenizer = load_tokenizer(tokenizer_path)
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None or sha256_file(tokenizer_path) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-020 frozen tokenizer verification failed.")

    cache_dir = artifact_dir / "cache"
    benchmark_index = cache_dir / "benchmarks" / "benchmark-ngrams.sqlite"
    benchmark_index_sha256 = copy_exp011_benchmark_index(
        Path(exp012_artifact_dir) / "cache" / "benchmarks" / "benchmark-ngrams.sqlite", benchmark_index
    )
    contamination_filter = NgramContaminationFilter(None, config.data.contamination_ngram_size, sqlite_path=benchmark_index)
    source_configs = {
        "fineweb": config.data,
        "fineweb_edu": replace(config.data, dataset_repo="HuggingFaceFW/fineweb-edu", dataset_config="default", dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    }
    source_counters = {source: {"scanned_documents": 0, "accepted_documents": 0, "rejected_documents": 0, "validation_documents_excluded": 0} for source in SOURCE_ORDER}
    documents = {
        source: _screened_train_documents(source_configs[source], cache_dir / source, contamination_filter, source_counters[source])
        for source in SOURCE_ORDER
    }
    mixer = GlobalDeduplicatedTokenMixer(documents, tokenizer, eod_id, EXP020_TARGETS, EXP020_STORED_TOKEN_IDS)
    stream_path = artifact_dir / "train-token-stream.uint16"
    stream = write_token_stream(stream_path, mixer, EXP020_STORED_TOKEN_IDS, config.data.context_length)
    if sum(mixer.prediction_token_contributions.values()) != EXP020_PREDICTION_TOKENS:
        raise RuntimeError("EXP-020 source contributions do not account for every prediction token.")
    observed_exp012 = sha256_file_prefix(stream_path, EXP012_PREFIX_BYTE_COUNT)
    if observed_exp012 != EXP012_STREAM_SHA256:
        raise RuntimeError(f"EXP-020 deterministic rebuild diverges from EXP-012 prefix: {observed_exp012}")
    observed_exp011 = sha256_file_prefix(stream_path, EXP011_PREFIX_BYTE_COUNT)
    if observed_exp011 != EXP011_STREAM_SHA256:
        raise RuntimeError(f"EXP-020 deterministic rebuild diverges from EXP-011 prefix: {observed_exp011}")

    general_values = torch.load(general_path, map_location="cpu", weights_only=True)
    edu_values = torch.load(edu_path, map_location="cpu", weights_only=True)
    manifest = {
        "experiment_id": "EXP-020",
        "preparation_mode": "full_stream",
        "preparation_strategy": "deterministic_full_rebuild_from_stream_zero_with_single_global_dedup_state; extension rejected because EXP-012 did not serialize dedup/cursor/mixture continuation state",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "mixture": {**config.mixture, "deterministic_mixing_method": "token-deficit-balanced whole-document selection; FineWeb wins deterministic ties", "actual_prediction_token_contributions": mixer.prediction_token_contributions, "actual_stored_token_contributions": mixer.stored_token_contributions, "documents_contributed": mixer.documents_contributed, "cross_source_duplicates_skipped": mixer.cross_source_duplicates_skipped, "intra_source_duplicates_skipped": mixer.intra_source_duplicates_skipped, "unique_document_count": len(mixer.selected_document_ids), "global_dedup_scope": "entire 7.2B artifact from stream zero"},
        "split": frozen_manifest["split"],
        "contamination": {"method": contamination["method"], "ngram_size": contamination["ngram_size"], "benchmark_sources": contamination["benchmark_sources"], "frozen_exp012_index_sha256": benchmark_index_sha256, "index_reused_byte_for_byte": True, "sources": source_counters},
        "frozen_exp012_source": {"manifest_sha256": EXP012_MANIFEST_SHA256, "stream_sha256": EXP012_STREAM_SHA256, "stored_token_ids": EXP012_STORED_TOKEN_IDS, "prediction_tokens": EXP012_PREDICTION_TOKENS},
        "exp012_prefix": {"byte_count": EXP012_PREFIX_BYTE_COUNT, "expected_sha256": EXP012_STREAM_SHA256, "observed_sha256": observed_exp012, "prefix_match": True, "stored_token_ids": EXP012_STORED_TOKEN_IDS},
        "exp011_prefix": dict(frozen_manifest["exp011_prefix"]),
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": sha256_file(tokenizer_path), "vocab_size": 8192, "special_tokens": [config.data.eod_token]},
        "general_validation": _validation_record(general_path, general_values),
        "edu_validation": {**_validation_record(edu_path, edu_values), "contamination_screened": True, "frozen_from": str(frozen_edu)},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": config.data.context_length, "prediction_tokens_per_example": config.training.sequence_predictions, "train_prediction_tokens": EXP020_PREDICTION_TOKENS, "train_token_count_including_final_target": EXP020_STORED_TOKEN_IDS, "train_examples": len(stream), "train_stream_file": stream_path.name, "train_stream_bytes": stream_path.stat().st_size, "train_stream_sha256": sha256_file(stream_path), "non_cycled": True},
        "build_command": "scripts/prepare_exp020.py --config configs/exp020-final-7p2b-cosine.yaml --artifact-dir ... --exp012-artifact-dir ... --recorded-source-commit ...",
        "builder_source_commit": recorded_source_commit,
        "builder_source": "deterministic rebuild from frozen EXP-012 tokenizer/validation/contamination provenance",
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    assert_exp020_prefix_provenance(manifest, stream_path)
    atomic_json_write(artifact_dir / "manifest.json", manifest)
    return manifest
