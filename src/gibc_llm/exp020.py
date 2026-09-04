"""EXP-020 deterministic 7.2B rebuild from stream zero with immutable prefix gates."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, replace
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


BUILD_STREAM_CHUNK_IDS = 1_048_576
BUILD_PROGRESS_INTERVAL_IDS = 10_000_000
NATIVE_SCRATCH_SAFETY_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class NativeScratch:
    root: Path
    cache_dir: Path
    benchmark_index: Path
    benchmark_index_sha256: str
    required_free_bytes: int
    available_free_bytes: int


class BuildProgressRecorder:
    # Output-only, durable telemetry for long deterministic data builds.

    def __init__(
        self,
        path: Path,
        target_stored_ids: int,
        mixer: GlobalDeduplicatedTokenMixer,
        source_counters: dict[str, dict[str, int]],
        *,
        started_at: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.target_stored_ids = target_stored_ids
        self.mixer = mixer
        self.source_counters = source_counters
        self.started_at = time.perf_counter() if started_at is None else started_at
        self._last_stored_ids: int | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stored_ids: int, *, now: float | None = None) -> None:
        if not 0 <= stored_ids <= self.target_stored_ids:
            raise ValueError("Progress stored-ID count is outside the frozen target range.")
        if stored_ids == self._last_stored_ids:
            return
        current = time.perf_counter() if now is None else now
        wall_seconds = max(0.0, current - self.started_at)
        emitted_ids_per_second = stored_ids / wall_seconds if wall_seconds else 0.0
        documents = sum(self.mixer.documents_contributed.values())
        payload = {
            "kind": "exp020_data_build_progress",
            "stored_ids_emitted": stored_ids,
            "prediction_tokens_emitted": max(0, stored_ids - 1),
            "percent_complete": 100.0 * stored_ids / self.target_stored_ids,
            "prediction_token_contributions": dict(self.mixer.prediction_token_contributions),
            "documents_contributed": dict(self.mixer.documents_contributed),
            "source_counters": {source: dict(values) for source, values in self.source_counters.items()},
            "intra_source_duplicate_skips": dict(self.mixer.intra_source_duplicates_skipped),
            "cross_source_duplicate_skips": dict(self.mixer.cross_source_duplicates_skipped),
            "unique_selected_documents": len(self.mixer.selected_document_ids),
            "wall_seconds": wall_seconds,
            "emitted_ids_per_second": emitted_ids_per_second,
            "documents_per_second": documents / wall_seconds if wall_seconds else 0.0,
            "estimated_remaining_seconds": (
                (self.target_stored_ids - stored_ids) / emitted_ids_per_second if emitted_ids_per_second else None
            ),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._last_stored_ids = stored_ids

    def close(self) -> None:
        return None


def stage_exp020_native_scratch(source_index: Path, scratch_dir: Path, artifact_label: str) -> NativeScratch:
    # Stage immutable hot build inputs on native WSL storage before construction.
    source_index = Path(source_index)
    if not source_index.is_file():
        raise RuntimeError("EXP-020 requires the immutable EXP-012 benchmark n-gram index.")
    root = Path(scratch_dir).resolve()
    if len(root.parts) >= 2 and root.parts[1] == "mnt":
        raise ValueError("EXP-020 hot scratch must be on native WSL storage, not /mnt.")
    root.mkdir(parents=True, exist_ok=True)
    required = EXP020_STORED_TOKEN_IDS * 2 + source_index.stat().st_size + NATIVE_SCRATCH_SAFETY_BYTES
    available = shutil.disk_usage(root).free
    if available < required:
        raise RuntimeError(
            f"EXP-020 native scratch requires at least {required} free bytes; observed {available}."
        )
    runtime_root = root / artifact_label
    if runtime_root.exists():
        raise RuntimeError("EXP-020 native scratch runtime already exists; choose a fresh empty scratch target.")
    cache_dir = runtime_root / "cache"
    benchmark_index = cache_dir / "benchmarks" / "benchmark-ngrams.sqlite"
    benchmark_index.parent.mkdir(parents=True, exist_ok=False)
    benchmark_index_sha256 = copy_exp011_benchmark_index(source_index, benchmark_index)
    return NativeScratch(
        root=runtime_root,
        cache_dir=cache_dir,
        benchmark_index=benchmark_index,
        benchmark_index_sha256=benchmark_index_sha256,
        required_free_bytes=required,
        available_free_bytes=available,
    )


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


def prepare_exp020(
    config: Any,
    artifact_dir: Path,
    exp012_artifact_dir: Path,
    recorded_source_commit: str,
    scratch_dir: Path,
    progress_interval_stored_ids: int = BUILD_PROGRESS_INTERVAL_IDS,
) -> dict[str, Any]:
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
    if progress_interval_stored_ids <= 0:
        raise ValueError("EXP-020 progress interval must be positive.")
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

    scratch = stage_exp020_native_scratch(
        Path(exp012_artifact_dir) / "cache" / "benchmarks" / "benchmark-ngrams.sqlite",
        scratch_dir,
        artifact_dir.name,
    )
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

    benchmark_index = scratch.benchmark_index
    benchmark_index_sha256 = scratch.benchmark_index_sha256
    contamination_filter = NgramContaminationFilter(None, config.data.contamination_ngram_size, sqlite_path=benchmark_index)
    source_configs = {
        "fineweb": config.data,
        "fineweb_edu": replace(config.data, dataset_repo="HuggingFaceFW/fineweb-edu", dataset_config="default", dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    }
    source_counters = {source: {"scanned_documents": 0, "accepted_documents": 0, "rejected_documents": 0, "validation_documents_excluded": 0} for source in SOURCE_ORDER}
    documents = {
        source: _screened_train_documents(source_configs[source], scratch.cache_dir / source, contamination_filter, source_counters[source])
        for source in SOURCE_ORDER
    }
    mixer = GlobalDeduplicatedTokenMixer(documents, tokenizer, eod_id, EXP020_TARGETS, EXP020_STORED_TOKEN_IDS)
    stream_path = artifact_dir / "train-token-stream.uint16"
    scratch_stream_path = scratch.root / stream_path.name
    recorder = BuildProgressRecorder(
        artifact_dir / "progress.jsonl",
        EXP020_STORED_TOKEN_IDS,
        mixer,
        source_counters,
        started_at=started,
    )
    next_progress = progress_interval_stored_ids

    def record_progress(stored_ids: int) -> None:
        nonlocal next_progress
        if stored_ids >= next_progress:
            recorder.record(stored_ids)
            while next_progress <= stored_ids:
                next_progress += progress_interval_stored_ids

    try:
        stream = write_token_stream(
            scratch_stream_path,
            mixer,
            EXP020_STORED_TOKEN_IDS,
            config.data.context_length,
            chunk_size_ids=BUILD_STREAM_CHUNK_IDS,
            progress_callback=record_progress,
        )
        recorder.record(EXP020_STORED_TOKEN_IDS)
    finally:
        recorder.close()
        contamination_filter.close()
    if sum(mixer.prediction_token_contributions.values()) != EXP020_PREDICTION_TOKENS:
        raise RuntimeError("EXP-020 source contributions do not account for every prediction token.")
    observed_exp012 = sha256_file_prefix(scratch_stream_path, EXP012_PREFIX_BYTE_COUNT)
    if observed_exp012 != EXP012_STREAM_SHA256:
        raise RuntimeError(f"EXP-020 deterministic rebuild diverges from EXP-012 prefix: {observed_exp012}")
    observed_exp011 = sha256_file_prefix(scratch_stream_path, EXP011_PREFIX_BYTE_COUNT)
    if observed_exp011 != EXP011_STREAM_SHA256:
        raise RuntimeError(f"EXP-020 deterministic rebuild diverges from EXP-011 prefix: {observed_exp011}")
    shutil.copyfile(scratch_stream_path, stream_path)
    if sha256_file(stream_path) != sha256_file(scratch_stream_path):
        raise RuntimeError("EXP-020 final stream copy diverged from its native-scratch materialization.")

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
        "operational_build": {"native_wsl_scratch": True, "native_scratch_required_free_bytes": scratch.required_free_bytes, "native_scratch_available_free_bytes_at_start": scratch.available_free_bytes, "sqlite_index_staged_byte_for_byte": True, "source_cache_on_native_scratch": True, "stream_materialized_on_native_scratch": True, "stream_serialization_chunk_ids": BUILD_STREAM_CHUNK_IDS, "durable_progress_interval_stored_ids": progress_interval_stored_ids},
        "frozen_exp012_source": {"manifest_sha256": EXP012_MANIFEST_SHA256, "stream_sha256": EXP012_STREAM_SHA256, "stored_token_ids": EXP012_STORED_TOKEN_IDS, "prediction_tokens": EXP012_PREDICTION_TOKENS},
        "exp012_prefix": {"byte_count": EXP012_PREFIX_BYTE_COUNT, "expected_sha256": EXP012_STREAM_SHA256, "observed_sha256": observed_exp012, "prefix_match": True, "stored_token_ids": EXP012_STORED_TOKEN_IDS},
        "exp011_prefix": dict(frozen_manifest["exp011_prefix"]),
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": sha256_file(tokenizer_path), "vocab_size": 8192, "special_tokens": [config.data.eod_token]},
        "general_validation": _validation_record(general_path, general_values),
        "edu_validation": {**_validation_record(edu_path, edu_values), "contamination_screened": True, "frozen_from": str(frozen_edu)},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": config.data.context_length, "prediction_tokens_per_example": config.training.sequence_predictions, "train_prediction_tokens": EXP020_PREDICTION_TOKENS, "train_token_count_including_final_target": EXP020_STORED_TOKEN_IDS, "train_examples": len(stream), "train_stream_file": stream_path.name, "train_stream_bytes": stream_path.stat().st_size, "train_stream_sha256": sha256_file(stream_path), "non_cycled": True},
        "build_command": "scripts/prepare_exp020.py --config configs/exp020-final-7p2b-cosine.yaml --artifact-dir ... --exp012-artifact-dir ... --scratch-dir ... --recorded-source-commit ...",
        "builder_source_commit": recorded_source_commit,
        "builder_source": "deterministic rebuild from frozen EXP-012 tokenizer/validation/contamination provenance",
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    assert_exp020_prefix_provenance(manifest, stream_path)
    atomic_json_write(artifact_dir / "manifest.json", manifest)
    return manifest
