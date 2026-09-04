"""Measure the source-faithful EXP-020 data builder without creating a final artifact."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from gibc_llm.data import NgramContaminationFilter, write_token_stream
from gibc_llm.exp003 import FROZEN_TOKENIZER_SHA256
from gibc_llm.exp004 import SOURCE_ORDER, GlobalDeduplicatedTokenMixer, _screened_train_documents
from gibc_llm.exp020 import (
    EXP020_PREDICTION_TOKENS,
    assert_frozen_exp012_source,
    stage_exp020_native_scratch,
)
from gibc_llm.tokenizer import load_tokenizer
from gibc_llm.utils import atomic_json_write, sha256_file


class TimedContaminationFilter:
    """Observational wrapper; delegates every source-faithful screening call unchanged."""

    def __init__(self, delegate: NgramContaminationFilter, timings: dict[str, float]) -> None:
        self.delegate = delegate
        self.timings = timings
        self.calls = 0

    def screen(self, text: str) -> Any:
        started = time.perf_counter()
        result = self.delegate.screen(text)
        self.timings["contamination_seconds"] += time.perf_counter() - started
        self.calls += 1
        return result

    def close(self) -> None:
        self.delegate.close()


class TimedTokenizer:
    """Observational tokenizer wrapper; encode outputs are delegated unchanged."""

    def __init__(self, delegate: Any, timings: dict[str, float]) -> None:
        self.delegate = delegate
        self.timings = timings
        self.calls = 0

    def encode(self, text: str) -> Any:
        started = time.perf_counter()
        result = self.delegate.encode(text)
        self.timings["tokenizer_seconds"] += time.perf_counter() - started
        self.calls += 1
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp020-final-7p2b-cosine.yaml"))
    parser.add_argument("--exp012-artifact-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--prediction-tokens", type=int, default=15_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from gibc_llm.utils import load_config
    import gibc_llm.exp004 as exp004_module

    if args.prediction_tokens <= 0 or args.prediction_tokens % 3:
        raise ValueError("Proxy prediction tokens must be a positive multiple of three for the frozen 2:1 ratio.")
    config = load_config(args.config)
    if (
        config.experiment_id != "EXP-020"
        or config.training.full_training_tokens != EXP020_PREDICTION_TOKENS
        or config.model.qk_norm
        or config.training.cautious_weight_decay
    ):
        raise RuntimeError("Proxy requires the frozen ordinary-AdamW EXP-020 configuration.")
    frozen_tokenizer, _, _, _, manifest = assert_frozen_exp012_source(args.exp012_artifact_dir)
    scratch = stage_exp020_native_scratch(
        Path(args.exp012_artifact_dir) / "cache" / "benchmarks" / "benchmark-ngrams.sqlite",
        args.scratch_dir,
        args.run_label,
    )
    tokenizer = load_tokenizer(frozen_tokenizer)
    if sha256_file(frozen_tokenizer) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("Frozen tokenizer identity changed before proxy.")
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("Frozen tokenizer is missing the EOD token.")

    targets = {"fineweb": args.prediction_tokens * 2 // 3, "fineweb_edu": args.prediction_tokens // 3}
    stored_ids = args.prediction_tokens + 1
    timings = {"source_fetch_seconds": 0.0, "contamination_seconds": 0.0, "tokenizer_seconds": 0.0}
    original_iterator = exp004_module.iter_fineweb_documents

    def timed_source_iterator(data_config: Any, revision: str, cache_dir: Path) -> Iterator[str]:
        iterator = iter(original_iterator(data_config, revision, cache_dir))
        while True:
            started = time.perf_counter()
            try:
                value = next(iterator)
            except StopIteration:
                timings["source_fetch_seconds"] += time.perf_counter() - started
                return
            timings["source_fetch_seconds"] += time.perf_counter() - started
            yield value

    source_configs = {
        "fineweb": config.data,
        "fineweb_edu": replace(
            config.data,
            dataset_repo="HuggingFaceFW/fineweb-edu",
            dataset_config="default",
            dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        ),
    }
    counters = {
        source: {
            "scanned_documents": 0,
            "accepted_documents": 0,
            "rejected_documents": 0,
            "validation_documents_excluded": 0,
        }
        for source in SOURCE_ORDER
    }
    contamination = TimedContaminationFilter(
        NgramContaminationFilter(None, config.data.contamination_ngram_size, sqlite_path=scratch.benchmark_index),
        timings,
    )
    timed_tokenizer = TimedTokenizer(tokenizer, timings)
    exp004_module.iter_fineweb_documents = timed_source_iterator
    started = time.perf_counter()
    try:
        documents = {
            source: _screened_train_documents(
                source_configs[source],
                scratch.cache_dir / source,
                contamination,
                counters[source],
            )
            for source in SOURCE_ORDER
        }
        mixer = GlobalDeduplicatedTokenMixer(documents, timed_tokenizer, eod_id, targets, stored_ids)
        stream_path = scratch.root / "proxy-token-stream.uint16"
        write_token_stream(stream_path, mixer, stored_ids, config.data.context_length, chunk_size_ids=1_048_576)
    finally:
        exp004_module.iter_fineweb_documents = original_iterator
        contamination.close()
    end_to_end_seconds = time.perf_counter() - started
    if sum(mixer.prediction_token_contributions.values()) != args.prediction_tokens:
        raise RuntimeError("Proxy mixture did not preserve exact 2:1 token accounting.")

    serializer_path = scratch.root / "serialization-only.uint16"
    serializer_started = time.perf_counter()
    write_token_stream(
        serializer_path,
        itertools.islice(itertools.cycle(range(8192)), stored_ids),
        stored_ids,
        config.data.context_length,
        chunk_size_ids=1_048_576,
    )
    serialization_seconds = time.perf_counter() - serializer_started
    documents = sum(mixer.documents_contributed.values())
    report = {
        "kind": "exp020_data_builder_non_scientific_proxy",
        "source_configuration": {
            "experiment_id": config.experiment_id,
            "config_path": str(args.config),
            "frozen_exp012_manifest_experiment_id": manifest["experiment_id"],
            "tokenizer_sha256": FROZEN_TOKENIZER_SHA256,
            "contamination_index_sha256": scratch.benchmark_index_sha256,
        },
        "proxy_prediction_tokens": args.prediction_tokens,
        "proxy_stored_ids": stored_ids,
        "proxy_stream_sha256": sha256_file(stream_path),
        "source_targets": targets,
        "actual_prediction_token_contributions": mixer.prediction_token_contributions,
        "documents_contributed": mixer.documents_contributed,
        "source_counters": counters,
        "duplicate_skips": {
            "intra_source": mixer.intra_source_duplicates_skipped,
            "cross_source": mixer.cross_source_duplicates_skipped,
        },
        "timings_seconds": {**timings, "end_to_end": end_to_end_seconds, "serialization_only": serialization_seconds},
        "rates": {
            "end_to_end_stored_ids_per_second": stored_ids / end_to_end_seconds,
            "accepted_prediction_tokens_per_second": args.prediction_tokens / end_to_end_seconds,
            "documents_screened_per_second": sum(item["scanned_documents"] for item in counters.values()) / end_to_end_seconds,
            "documents_accepted_per_second": sum(item["accepted_documents"] for item in counters.values()) / end_to_end_seconds,
            "contamination_documents_per_second": contamination.calls / timings["contamination_seconds"] if timings["contamination_seconds"] else 0.0,
            "tokenizer_documents_per_second": timed_tokenizer.calls / timings["tokenizer_seconds"] if timings["tokenizer_seconds"] else 0.0,
            "serialization_only_ids_per_second": stored_ids / serialization_seconds,
        },
        "operational_note": "Non-scientific systems proxy only; no final stream, prefix proof, model training, validation, or benchmark evaluation.",
        "scratch": {
            "root": str(scratch.root),
            "required_free_bytes_for_final": scratch.required_free_bytes,
            "available_free_bytes_at_start": scratch.available_free_bytes,
        },
    }
    atomic_json_write(args.report_path, report)


if __name__ == "__main__":
    main()
