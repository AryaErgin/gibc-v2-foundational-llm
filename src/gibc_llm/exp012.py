"""EXP-012 preparation: deterministic 2.4B full rebuild with hard EXP-011/006/004 prefix gates."""

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
from .exp004 import SOURCE_ORDER, GlobalDeduplicatedTokenMixer, _screened_train_documents, _validation_record, assert_frozen_exp004_artifacts
from .exp006 import EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256, EXP004_PREFIX_STORED_TOKEN_IDS, verify_stream_prefix
from .exp011 import EXP006_PREDICTION_TOKENS, EXP006_PREFIX_BYTE_COUNT
from .utils import atomic_json_write, sha256_file, sha256_file_prefix


EXP011_PREDICTION_TOKENS = 1_500_020_736
EXP011_STORED_TOKEN_IDS = EXP011_PREDICTION_TOKENS + 1
EXP011_PREFIX_BYTE_COUNT = EXP011_STORED_TOKEN_IDS * 2
EXP011_STREAM_SHA256 = "092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671"
EXP011_MANIFEST_SHA256 = "b2ed5e461d753beb581c0d88668371c16abc63c6c9a67673f453a46f27d9feeb"
EXP012_TARGETS = {"fineweb": 1_599_995_904, "fineweb_edu": 799_997_952}


def verify_exp011_prefix(stream_path: Path, byte_count: int, expected_sha256: str = EXP011_STREAM_SHA256) -> str:
    """Hash the exact verified 1.5B raw prefix and reject any divergence."""
    observed = sha256_file_prefix(stream_path, byte_count)
    if observed != expected_sha256:
        raise RuntimeError(
            f"EXP-012 EXP-011 prefix SHA-256 mismatch: expected {expected_sha256}, observed {observed}. "
            "The artifact is not authorized for training."
        )
    return observed


def copy_exp011_benchmark_index(source_path: Path, target_path: Path) -> str:
    """Copy the approved existing contamination index byte-for-byte; never rebuild it."""
    source, target = Path(source_path), Path(target_path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError("EXP-012 requires the existing nonempty EXP-011 contamination index.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_sha256 = sha256_file(source)
    if sha256_file(target) != source_sha256:
        raise RuntimeError("EXP-012 copied contamination index differs from the approved EXP-011 source bytes.")
    return source_sha256


def assert_frozen_exp011_source(exp011_artifact_dir: Path) -> tuple[Path, Path, Path, Path, dict[str, Any], str]:
    """Return only the independently verified immutable EXP-011 full-stream controls."""
    source = Path(exp011_artifact_dir)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != EXP011_MANIFEST_SHA256:
        raise RuntimeError("EXP-012 requires the exact audited EXP-011 manifest SHA-256.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = manifest.get("packed", {})
    stream = source / packed.get("train_stream_file", "train-token-stream.uint16")
    if (
        manifest.get("experiment_id") != "EXP-011"
        or packed.get("train_prediction_tokens") != EXP011_PREDICTION_TOKENS
        or packed.get("train_token_count_including_final_target") != EXP011_STORED_TOKEN_IDS
        or packed.get("train_stream_bytes") != EXP011_PREFIX_BYTE_COUNT
        or packed.get("train_stream_sha256") != EXP011_STREAM_SHA256
        or not stream.is_file()
        or stream.stat().st_size != EXP011_PREFIX_BYTE_COUNT
        or sha256_file(stream) != EXP011_STREAM_SHA256
    ):
        raise RuntimeError("EXP-012 requires the exact audited EXP-011 1.5B token stream.")
    exp006_prefix = manifest.get("exp006_prefix", {})
    frozen_exp006 = manifest.get("frozen_exp006_source", {})
    if (
        exp006_prefix.get("byte_count") != EXP006_PREFIX_BYTE_COUNT
        or not exp006_prefix.get("expected_sha256")
        or exp006_prefix.get("observed_sha256") != exp006_prefix.get("expected_sha256")
        or exp006_prefix.get("prefix_match") is not True
        or frozen_exp006.get("stream_sha256") != exp006_prefix.get("expected_sha256")
        or frozen_exp006.get("prediction_tokens") != EXP006_PREDICTION_TOKENS
        or verify_stream_prefix(stream, EXP006_PREFIX_BYTE_COUNT, exp006_prefix["expected_sha256"]) != exp006_prefix["expected_sha256"]
    ):
        raise RuntimeError("EXP-012 requires EXP-011's independently verified exact EXP-006 prefix chain.")
    exp004_prefix = manifest.get("exp004_prefix", {})
    if (
        exp004_prefix.get("byte_count") != EXP004_PREFIX_BYTE_COUNT
        or exp004_prefix.get("expected_sha256") != EXP004_PREFIX_SHA256
        or exp004_prefix.get("observed_sha256") != EXP004_PREFIX_SHA256
        or exp004_prefix.get("prefix_match") is not True
        or verify_stream_prefix(stream, EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256) != EXP004_PREFIX_SHA256
    ):
        raise RuntimeError("EXP-012 requires EXP-011's independently verified exact EXP-004 prefix chain.")
    tokenizer = source / "tokenizer" / "tokenizer.json"
    general, edu = source / "general_validation.pt", source / "edu_validation.pt"
    assert_frozen_exp004_artifacts(tokenizer, general, edu)
    if sha256_file(tokenizer) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-012 source tokenizer is not the frozen 8192-entry tokenizer.")
    return tokenizer, general, edu, stream, manifest, EXP011_MANIFEST_SHA256


def assert_exp012_prefix_provenance(manifest: dict[str, Any], stream_path: Path) -> None:
    """Independently recheck the immutable EXP-011/006/004 prefix chain before training."""
    frozen_exp011 = manifest.get("frozen_exp011_source", {})
    exp011_prefix = manifest.get("exp011_prefix", {})
    if (
        frozen_exp011.get("stream_sha256") != EXP011_STREAM_SHA256
        or frozen_exp011.get("manifest_sha256") != EXP011_MANIFEST_SHA256
        or frozen_exp011.get("stored_token_ids") != EXP011_STORED_TOKEN_IDS
        or frozen_exp011.get("prediction_tokens") != EXP011_PREDICTION_TOKENS
        or frozen_exp011.get("source_manifest_experiment_id") != "EXP-011"
        or exp011_prefix.get("byte_count") != EXP011_PREFIX_BYTE_COUNT
        or exp011_prefix.get("expected_sha256") != EXP011_STREAM_SHA256
        or exp011_prefix.get("observed_sha256") != EXP011_STREAM_SHA256
        or exp011_prefix.get("prefix_match") is not True
        or verify_exp011_prefix(stream_path, EXP011_PREFIX_BYTE_COUNT) != EXP011_STREAM_SHA256
    ):
        raise RuntimeError("EXP-012 manifest lacks a verified exact EXP-011 1.5B prefix.")
    exp006_prefix = manifest.get("exp006_prefix", {})
    if (
        exp006_prefix.get("byte_count") != EXP006_PREFIX_BYTE_COUNT
        or not exp006_prefix.get("expected_sha256")
        or exp006_prefix.get("observed_sha256") != exp006_prefix.get("expected_sha256")
        or exp006_prefix.get("prefix_match") is not True
        or verify_stream_prefix(stream_path, EXP006_PREFIX_BYTE_COUNT, exp006_prefix["expected_sha256"]) != exp006_prefix["expected_sha256"]
    ):
        raise RuntimeError("EXP-012 manifest lacks a verified exact EXP-006 900M prefix.")
    exp004_prefix = manifest.get("exp004_prefix", {})
    if (
        exp004_prefix.get("byte_count") != EXP004_PREFIX_BYTE_COUNT
        or exp004_prefix.get("expected_sha256") != EXP004_PREFIX_SHA256
        or exp004_prefix.get("observed_sha256") != EXP004_PREFIX_SHA256
        or exp004_prefix.get("prefix_match") is not True
        or verify_stream_prefix(stream_path, EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256) != EXP004_PREFIX_SHA256
    ):
        raise RuntimeError("EXP-012 manifest lacks a verified exact EXP-004 300M prefix.")


def prepare_exp012(config: Any, artifact_dir: Path, exp011_artifact_dir: Path) -> dict[str, Any]:
    """Rebuild one global 2.4B stream from zero and prove all immutable prefixes; never trains."""
    from .tokenizer import load_tokenizer

    if config.experiment_id != "EXP-012" or config.mixture is None or config.mixture.get("target_prediction_tokens") != EXP012_TARGETS:
        raise ValueError("prepare_exp012 requires the approved configs/exp012.yaml 2.4B 2:1 mixture specification.")
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        raise RuntimeError("EXP-012 artifact directory already exists; preserve it and choose a new directory rather than overwriting a build.")
    started = time.perf_counter()
    frozen_tokenizer, frozen_general, frozen_edu, frozen_stream, frozen_manifest, frozen_manifest_sha = assert_frozen_exp011_source(exp011_artifact_dir)
    frozen_contamination = frozen_manifest.get("contamination", {})
    frozen_benchmark_sources = frozen_contamination.get("benchmark_sources")
    if (
        frozen_contamination.get("method") != "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap"
        or frozen_contamination.get("ngram_size") != config.data.contamination_ngram_size
        or not isinstance(frozen_benchmark_sources, list)
        or not frozen_benchmark_sources
    ):
        raise RuntimeError("EXP-012 requires EXP-011's exact recorded contamination-screening provenance.")

    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "tokenizer").mkdir()
    tokenizer_path, general_path, edu_path = artifact_dir / "tokenizer" / "tokenizer.json", artifact_dir / "general_validation.pt", artifact_dir / "edu_validation.pt"
    shutil.copy2(frozen_tokenizer, tokenizer_path)
    shutil.copy2(frozen_general, general_path)
    shutil.copy2(frozen_edu, edu_path)
    if sha256_file(tokenizer_path) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-012 copied tokenizer differs from the frozen tokenizer hash.")
    tokenizer = load_tokenizer(tokenizer_path)
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("Frozen tokenizer does not contain the required EOD token.")

    cache_dir = artifact_dir / "cache"
    source_index = Path(exp011_artifact_dir) / "cache" / "benchmarks" / "benchmark-ngrams.sqlite"
    benchmark_index_path = cache_dir / "benchmarks" / "benchmark-ngrams.sqlite"
    benchmark_index_sha256 = copy_exp011_benchmark_index(source_index, benchmark_index_path)
    contamination_filter = NgramContaminationFilter(None, config.data.contamination_ngram_size, sqlite_path=benchmark_index_path)
    source_configs = {
        "fineweb": config.data,
        "fineweb_edu": replace(config.data, dataset_repo="HuggingFaceFW/fineweb-edu", dataset_config="default", dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    }
    source_counters = {source: {"scanned_documents": 0, "accepted_documents": 0, "rejected_documents": 0, "validation_documents_excluded": 0} for source in SOURCE_ORDER}
    documents = {source: _screened_train_documents(source_configs[source], cache_dir / source, contamination_filter, source_counters[source]) for source in SOURCE_ORDER}
    stored_tokens = config.training.full_training_tokens + 1
    mixer = GlobalDeduplicatedTokenMixer(documents, tokenizer, eod_id, EXP012_TARGETS, stored_tokens)
    stream_path = artifact_dir / "train-token-stream.uint16"
    stream = write_token_stream(stream_path, mixer, stored_tokens, config.data.context_length)
    if sum(mixer.prediction_token_contributions.values()) != config.training.full_training_tokens:
        raise RuntimeError("EXP-012 source contributions do not account for every prediction token exactly once.")
    observed_exp011 = verify_exp011_prefix(stream_path, EXP011_PREFIX_BYTE_COUNT)
    source_exp006_prefix = frozen_manifest["exp006_prefix"]
    observed_exp006 = verify_stream_prefix(stream_path, EXP006_PREFIX_BYTE_COUNT, source_exp006_prefix["expected_sha256"])
    observed_exp004 = verify_stream_prefix(stream_path, EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256)

    general_values, edu_values = torch.load(general_path, map_location="cpu", weights_only=True), torch.load(edu_path, map_location="cpu", weights_only=True)
    manifest = {
        "experiment_id": "EXP-012",
        "preparation_mode": "full_stream",
        "preparation_strategy": "deterministic_full_rebuild_from_stream_zero_with_single_global_dedup_state",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "mixture": {**config.mixture, "deterministic_mixing_method": "token-deficit-balanced whole-document selection; FineWeb wins deterministic ties", "actual_prediction_token_contributions": mixer.prediction_token_contributions, "actual_stored_token_contributions": mixer.stored_token_contributions, "documents_contributed": mixer.documents_contributed, "cross_source_duplicates_skipped": mixer.cross_source_duplicates_skipped, "intra_source_duplicates_skipped": mixer.intra_source_duplicates_skipped, "unique_document_count": len(mixer.selected_document_ids), "global_dedup_scope": "entire 2.4B artifact, including EXP-004/EXP-006/EXP-011 prefixes and extension"},
        "split": {"method": "sha256(seed:canonical_content_sha256) modulo buckets", "seed": config.data.split_seed, "validation_buckets": config.data.validation_bucket_cutoff, "modulus": config.data.validation_bucket_modulus},
        "contamination": {"method": "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap", "ngram_size": config.data.contamination_ngram_size, "benchmark_sources": frozen_benchmark_sources, "frozen_exp011_index_sha256": benchmark_index_sha256, "index_reused_byte_for_byte": True, "sources": source_counters},
        "frozen_exp011_source": {"stream_sha256": EXP011_STREAM_SHA256, "manifest_sha256": frozen_manifest_sha, "stored_token_ids": EXP011_STORED_TOKEN_IDS, "prediction_tokens": EXP011_PREDICTION_TOKENS, "source_manifest_experiment_id": frozen_manifest["experiment_id"]},
        "exp011_prefix": {"byte_count": EXP011_PREFIX_BYTE_COUNT, "expected_sha256": EXP011_STREAM_SHA256, "observed_sha256": observed_exp011, "prefix_match": True, "stored_token_ids": EXP011_STORED_TOKEN_IDS},
        "exp006_prefix": {"byte_count": EXP006_PREFIX_BYTE_COUNT, "expected_sha256": source_exp006_prefix["expected_sha256"], "observed_sha256": observed_exp006, "prefix_match": True, "stored_token_ids": EXP006_PREDICTION_TOKENS + 1},
        "exp004_prefix": {"byte_count": EXP004_PREFIX_BYTE_COUNT, "expected_sha256": EXP004_PREFIX_SHA256, "observed_sha256": observed_exp004, "prefix_match": True, "stored_token_ids": EXP004_PREFIX_STORED_TOKEN_IDS},
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": sha256_file(tokenizer_path), "vocab_size": 8192, "special_tokens": [config.data.eod_token]},
        "general_validation": _validation_record(general_path, general_values),
        "edu_validation": {**_validation_record(edu_path, edu_values), "contamination_screened": True, "frozen_from": str(frozen_edu)},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": config.data.context_length, "prediction_tokens_per_example": config.training.sequence_predictions, "train_prediction_tokens": config.training.full_training_tokens, "train_token_count_including_final_target": stored_tokens, "train_examples": len(stream), "train_stream_file": stream_path.name, "train_stream_bytes": stream_path.stat().st_size, "train_stream_sha256": sha256_file(stream_path), "non_cycled": True},
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    assert_exp012_prefix_provenance(manifest, stream_path)
    atomic_json_write(artifact_dir / "manifest.json", manifest)
    return manifest
