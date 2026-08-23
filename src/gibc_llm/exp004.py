"""EXP-004 deterministic globally deduplicated FineWeb/FineWeb-Edu mixture preparation."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from .data import (
    Document,
    assign_split,
    build_benchmark_filter,
    iter_fineweb_documents,
    resolve_fineweb_revision,
    stable_document_id,
    tensor_sha256,
    write_token_stream,
)
from .exp003 import (
    EDU_VALIDATION_PREDICTION_TOKENS,
    FROZEN_TOKENIZER_SHA256,
    GENERAL_VALIDATION_INPUTS_SHA256,
    GENERAL_VALIDATION_TARGETS_SHA256,
)
from .utils import atomic_json_write, sha256_file

EDU_VALIDATION_INPUTS_SHA256 = "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a"
EDU_VALIDATION_TARGETS_SHA256 = "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4"
SOURCE_ORDER = ("fineweb", "fineweb_edu")


class GlobalDeduplicatedTokenMixer:
    """Emit a deterministic source-balanced token stream while selecting every content ID once."""

    def __init__(
        self,
        sources: dict[str, Iterator[Document]],
        tokenizer: Any,
        eod_id: int,
        target_prediction_tokens: dict[str, int],
        stored_token_count: int,
    ) -> None:
        if tuple(sources) != SOURCE_ORDER or set(target_prediction_tokens) != set(SOURCE_ORDER):
            raise ValueError("EXP-004 requires deterministic FineWeb then FineWeb-Edu source identities.")
        if stored_token_count != sum(target_prediction_tokens.values()) + 1:
            raise ValueError("EXP-004 stored-token count must be prediction tokens plus exactly one final target.")
        self.sources = sources
        self.tokenizer = tokenizer
        self.eod_id = eod_id
        self.target_prediction_tokens = target_prediction_tokens
        self.stored_token_count = stored_token_count
        self.prediction_token_contributions = {name: 0 for name in SOURCE_ORDER}
        self.stored_token_contributions = {name: 0 for name in SOURCE_ORDER}
        self.documents_contributed = {name: 0 for name in SOURCE_ORDER}
        self.cross_source_duplicates_skipped = {name: 0 for name in SOURCE_ORDER}
        self.intra_source_duplicates_skipped = {name: 0 for name in SOURCE_ORDER}
        self.selected_document_ids: set[str] = set()
        self._owner_by_document_id: dict[str, str] = {}

    def _next_unique_document(self, source: str) -> Document:
        while True:
            try:
                document = next(self.sources[source])
            except StopIteration as error:
                raise RuntimeError(f"EXP-004 exhausted unique {source} documents before filling its exact stream.") from error
            owner = self._owner_by_document_id.get(document.document_id)
            if owner is not None:
                if owner == source:
                    self.intra_source_duplicates_skipped[source] += 1
                else:
                    self.cross_source_duplicates_skipped[source] += 1
                continue
            self._owner_by_document_id[document.document_id] = source
            self.selected_document_ids.add(document.document_id)
            self.documents_contributed[source] += 1
            return document

    def _choose_source(self, prediction_tokens_emitted: int) -> str:
        if prediction_tokens_emitted == 0:
            return "fineweb"
        total_target = sum(self.target_prediction_tokens.values())
        deficits = {
            source: self.target_prediction_tokens[source] * prediction_tokens_emitted / total_target
            - self.prediction_token_contributions[source]
            for source in SOURCE_ORDER
        }
        return max(SOURCE_ORDER, key=lambda source: (deficits[source], -SOURCE_ORDER.index(source)))

    def __iter__(self) -> Iterator[int]:
        stored_emitted = 0
        while stored_emitted < self.stored_token_count:
            source = self._choose_source(stored_emitted - 1 if stored_emitted else 0)
            document = self._next_unique_document(source)
            token_ids = [*self.tokenizer.encode(document.text).ids, self.eod_id]
            for token_id in token_ids:
                if stored_emitted >= self.stored_token_count:
                    return
                if stored_emitted > 0:
                    self.prediction_token_contributions[source] += 1
                self.stored_token_contributions[source] += 1
                stored_emitted += 1
                yield token_id


def assert_frozen_exp004_artifacts(tokenizer_path: Path, general_validation_path: Path, edu_validation_path: Path) -> None:
    """Reject any changed tokenizer or either frozen validation control."""
    if not Path(tokenizer_path).is_file() or sha256_file(tokenizer_path) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-004 requires the exact frozen tokenizer artifact.")
    for path, expected_inputs, expected_targets, label in (
        (general_validation_path, GENERAL_VALIDATION_INPUTS_SHA256, GENERAL_VALIDATION_TARGETS_SHA256, "general"),
        (edu_validation_path, EDU_VALIDATION_INPUTS_SHA256, EDU_VALIDATION_TARGETS_SHA256, "educational"),
    ):
        if not Path(path).is_file():
            raise RuntimeError(f"EXP-004 requires the frozen {label} validation artifact.")
        values = torch.load(path, map_location="cpu", weights_only=True)
        if tensor_sha256(values["inputs"]) != expected_inputs or tensor_sha256(values["targets"]) != expected_targets:
            raise RuntimeError(f"EXP-004 requires the exact frozen {label} validation hashes.")


def _screened_train_documents(
    data_config: Any,
    cache_dir: Path,
    contamination_filter: Any,
    counters: dict[str, int],
) -> Iterator[Document]:
    revision = resolve_fineweb_revision(data_config)
    for text in iter_fineweb_documents(data_config, revision, cache_dir):
        counters["scanned_documents"] += 1
        document_id = stable_document_id(text)
        decision = contamination_filter.screen(text)
        if decision.rejected:
            counters["rejected_documents"] += 1
            continue
        counters["accepted_documents"] += 1
        split = assign_split(
            document_id,
            data_config.split_seed,
            data_config.validation_bucket_modulus,
            data_config.validation_bucket_cutoff,
        )
        if split != "train":
            counters["validation_documents_excluded"] += 1
            continue
        yield Document(document_id, text, split)


def _validation_record(path: Path, values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "file": path.name,
        "prediction_tokens": int(values["targets"].numel()),
        "examples": int(values["inputs"].shape[0]),
        "inputs_sha256": tensor_sha256(values["inputs"]),
        "targets_sha256": tensor_sha256(values["targets"]),
    }


def prepare_exp004(config: Any, artifact_dir: Path, exp001_artifact_dir: Path, exp003_artifact_dir: Path) -> dict[str, Any]:
    """Materialize the exact non-cycled globally deduplicated 2:1 token mixture; never trains."""
    from .tokenizer import load_tokenizer

    if config.experiment_id != "EXP-004" or config.mixture is None:
        raise ValueError("prepare_exp004 requires the approved configs/exp004.yaml mixture specification.")
    started = time.perf_counter()
    exp001 = Path(exp001_artifact_dir)
    exp003 = Path(exp003_artifact_dir)
    tokenizer_path = exp001 / "tokenizer" / "tokenizer.json"
    general_validation_path = exp001 / "validation.pt"
    edu_validation_path = exp003 / "edu_validation.pt"
    assert_frozen_exp004_artifacts(tokenizer_path, general_validation_path, edu_validation_path)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "tokenizer").mkdir(exist_ok=True)
    copied_tokenizer = artifact_dir / "tokenizer" / "tokenizer.json"
    copied_general = artifact_dir / "general_validation.pt"
    copied_edu = artifact_dir / "edu_validation.pt"
    shutil.copy2(tokenizer_path, copied_tokenizer)
    shutil.copy2(general_validation_path, copied_general)
    shutil.copy2(edu_validation_path, copied_edu)
    tokenizer = load_tokenizer(tokenizer_path)
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("Frozen tokenizer does not contain the required EOD token.")

    cache_dir = artifact_dir / "cache"
    contamination_filter, benchmark_sources = build_benchmark_filter(
        cache_dir / "benchmarks", config.data.contamination_ngram_size
    )
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
    source_documents = {
        source: _screened_train_documents(source_configs[source], cache_dir / source, contamination_filter, source_counters[source])
        for source in SOURCE_ORDER
    }
    target_prediction_tokens = dict(config.mixture["target_prediction_tokens"])
    stored_tokens = config.training.full_training_tokens + 1
    mixer = GlobalDeduplicatedTokenMixer(source_documents, tokenizer, eod_id, target_prediction_tokens, stored_tokens)
    stream_path = artifact_dir / "train-token-stream.uint16"
    stream = write_token_stream(stream_path, mixer, stored_tokens, config.data.context_length)
    if sum(mixer.prediction_token_contributions.values()) != config.training.full_training_tokens:
        raise RuntimeError("EXP-004 mixture source contributions do not account for every prediction token exactly once.")

    general_values = torch.load(copied_general, map_location="cpu", weights_only=True)
    edu_values = torch.load(copied_edu, map_location="cpu", weights_only=True)
    manifest = {
        "experiment_id": "EXP-004",
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
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": sha256_file(copied_tokenizer), "vocab_size": 8192, "special_tokens": [config.data.eod_token]},
        "general_validation": _validation_record(copied_general, general_values),
        "edu_validation": {**_validation_record(copied_edu, edu_values), "contamination_screened": True, "frozen_from": str(edu_validation_path)},
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
