"""Source-faithful systems tests for the EXP-020 deterministic data builder."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import shutil
from pathlib import Path

import numpy as np

from gibc_llm.data import Document, NgramContaminationFilter, stable_document_id, write_token_stream


def _create_index(path: Path, values: set[bytes]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE ngram_hashes (value BLOB PRIMARY KEY)")
        connection.executemany("INSERT INTO ngram_hashes(value) VALUES (?)", [(value,) for value in values])


def _legacy_write(path: Path, token_ids: list[int]) -> None:
    values = np.memmap(path, mode="w+", dtype=np.uint16, shape=(len(token_ids),))
    try:
        for index, token_id in enumerate(token_ids):
            values[index] = token_id
        values.flush()
    finally:
        del values


def test_sqlite_screening_reuses_one_read_only_connection_and_matches_reference(monkeypatch, tmp_path: Path) -> None:
    """Breaks if EXP-020 opens a SQLite connection per document or changes a decision."""
    from gibc_llm import data as data_module

    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    texts = ("orchards quietly measure copper meteor trails beneath glass lanterns each evening", f"prefix {benchmark} suffix")
    reference = NgramContaminationFilter.from_texts([benchmark], ngram_size=13)
    index = tmp_path / "benchmark-ngrams.sqlite"
    _create_index(index, reference.ngram_hashes or set())
    original_connect, calls = sqlite3.connect, []

    def tracked_connect(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(data_module.sqlite3, "connect", tracked_connect)
    screening = NgramContaminationFilter(None, ngram_size=13, sqlite_path=index)
    assert [screening.screen(text).as_dict() for text in texts] == [reference.screen(text).as_dict() for text in texts]
    assert len(calls) == 1
    assert calls[0][1]["uri"] is True
    assert "mode=ro" in str(calls[0][0][0])
    screening.close()
    assert screening.sqlite_connection is None


def test_chunked_stream_writer_is_an_explicit_bounded_serialization_path() -> None:
    """Breaks if the writer still has only the old per-token memmap API."""
    signature = inspect.signature(write_token_stream)
    assert "chunk_size_ids" in signature.parameters
    assert "progress_callback" in signature.parameters


def test_chunked_stream_writer_preserves_bytes_and_progress_at_buffer_boundaries(tmp_path: Path) -> None:
    """Breaks if a buffer boundary changes IDs, truncation, or emitted-count reporting."""
    token_ids = [8191, 0, 3, 512, 7, 11, 0, 13, 17, 19, 23]
    legacy_path, chunked_path = tmp_path / "legacy.uint16", tmp_path / "chunked.uint16"
    _legacy_write(legacy_path, token_ids)
    progress: list[int] = []
    write_token_stream(chunked_path, iter(token_ids), len(token_ids), 2, chunk_size_ids=3, progress_callback=progress.append)
    assert chunked_path.read_bytes() == legacy_path.read_bytes()
    assert progress == [3, 6, 9, 11]


def test_exp020_progress_records_accounting_without_mutating_the_mixer(tmp_path: Path) -> None:
    """Breaks if durable telemetry is absent or changes already-emitted accounting."""
    from gibc_llm.exp020 import BuildProgressRecorder

    class Mixer:
        prediction_token_contributions = {"fineweb": 8, "fineweb_edu": 4}
        documents_contributed = {"fineweb": 2, "fineweb_edu": 1}
        intra_source_duplicates_skipped = {"fineweb": 1, "fineweb_edu": 0}
        cross_source_duplicates_skipped = {"fineweb": 0, "fineweb_edu": 1}
        selected_document_ids = {"a", "b", "c"}
    counters = {"fineweb": {"scanned_documents": 3, "accepted_documents": 2, "rejected_documents": 1, "validation_documents_excluded": 0}, "fineweb_edu": {"scanned_documents": 2, "accepted_documents": 1, "rejected_documents": 0, "validation_documents_excluded": 1}}
    mixer = Mixer()
    recorder = BuildProgressRecorder(tmp_path / "progress.jsonl", 13, mixer, counters, started_at=100.0)
    recorder.record(13, now=104.0)
    recorder.close()
    record = json.loads((tmp_path / "progress.jsonl").read_text(encoding="utf-8"))
    assert (record["stored_ids_emitted"], record["prediction_tokens_emitted"], record["percent_complete"]) == (13, 12, 100.0)
    assert record["prediction_token_contributions"] == mixer.prediction_token_contributions
    assert record["source_counters"] == counters


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _Tokenizer:
    def encode(self, text: str) -> _Encoded:
        values = {"a": [101, 102, 103], "c": [201, 202, 203], "d": [301, 302, 303], "e": [401, 402, 403]}
        return _Encoded(values[text])


def _document(text: str) -> Document:
    return Document(stable_document_id(text), text, "train")


def test_chunked_writer_preserves_global_mixer_semantics_across_dedup_switch_and_cutoff(tmp_path: Path) -> None:
    # The old and new writers receive identical frozen mixer inputs and must emit
    # identical bytes despite source switching, duplicate skips, EODs, and a
    # final mid-document cutoff.
    from gibc_llm.exp004 import GlobalDeduplicatedTokenMixer

    def make_mixer() -> GlobalDeduplicatedTokenMixer:
        return GlobalDeduplicatedTokenMixer(
            {
                "fineweb": iter([_document("a"), _document("a"), _document("c"), _document("d")]),
                "fineweb_edu": iter([_document("a"), _document("e")]),
            },
            _Tokenizer(),
            eod_id=99,
            target_prediction_tokens={"fineweb": 8, "fineweb_edu": 4},
            stored_token_count=13,
        )

    legacy, optimized = make_mixer(), make_mixer()
    legacy_path, optimized_path = tmp_path / "legacy-mixed.uint16", tmp_path / "optimized-mixed.uint16"
    legacy_ids = list(legacy)
    _legacy_write(legacy_path, legacy_ids)
    write_token_stream(optimized_path, optimized, 13, context_length=2, chunk_size_ids=3)
    assert optimized_path.read_bytes() == legacy_path.read_bytes()
    assert hashlib.sha256(optimized_path.read_bytes()).hexdigest() == hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    assert optimized.prediction_token_contributions == legacy.prediction_token_contributions == {"fineweb": 8, "fineweb_edu": 4}
    assert optimized.intra_source_duplicates_skipped == legacy.intra_source_duplicates_skipped == {"fineweb": 1, "fineweb_edu": 0}
    assert optimized.cross_source_duplicates_skipped == legacy.cross_source_duplicates_skipped == {"fineweb": 0, "fineweb_edu": 1}


def test_sqlite_fast_path_preserves_clean_rejection_and_normalization_decisions(tmp_path: Path) -> None:
    # Verifies the source-faithful NFKC/casefold 13-gram membership decision,
    # including a clean document and a normalization-equivalent contaminated one.
    benchmark = "ONE two three four five six seven eight nine ten eleven twelve thirteen"
    normalized_match = "one TWO three four five six seven eight nine ten eleven twelve thirteen"
    clean = "unrelated materials remain clean despite accents and fullwidth forms"
    reference = NgramContaminationFilter.from_texts([benchmark], ngram_size=13)
    index = tmp_path / "index.sqlite"
    _create_index(index, reference.ngram_hashes or set())
    fast = NgramContaminationFilter(None, ngram_size=13, sqlite_path=index)
    assert fast.screen(clean).as_dict() == reference.screen(clean).as_dict()
    assert fast.screen(normalized_match).as_dict() == reference.screen(normalized_match).as_dict()
    fast.close()


def test_native_scratch_stages_only_a_byte_identical_index_and_rejects_drvfs(tmp_path: Path) -> None:
    # The fast path may move hot I/O, but may not alter the immutable index bytes.
    import pytest
    from gibc_llm.exp020 import stage_exp020_native_scratch

    source_index = tmp_path / "source.sqlite"
    _create_index(source_index, {b"x" * 32})
    # pytest's /tmp is a 16-GB tmpfs, intentionally below the full-build gate;
    # use a unique child of the ext4 repository filesystem for this staging test.
    native_root = Path.cwd() / ".pytest-exp020-native-scratch" / tmp_path.name
    try:
        scratch = stage_exp020_native_scratch(source_index, native_root, "runtime")
        assert scratch.root.parts[1] != "mnt"
        assert scratch.benchmark_index.read_bytes() == source_index.read_bytes()
        assert scratch.benchmark_index_sha256 == hashlib.sha256(source_index.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="native WSL"):
            stage_exp020_native_scratch(source_index, Path("/mnt/c/forbidden-exp020-scratch"), "runtime")
    finally:
        shutil.rmtree(native_root, ignore_errors=True)


def test_exp020_ram_index_mode_is_exact_and_sqlite_remains_default(tmp_path: Path) -> None:
    """Breaks if the operational RAM mode changes complete decisions or default behavior."""
    import pytest
    from gibc_llm.exp020 import make_exp020_contamination_filter

    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    reference = NgramContaminationFilter.from_texts([benchmark], ngram_size=13)
    index = tmp_path / "index.sqlite"
    _create_index(index, reference.ngram_hashes or set())
    default = make_exp020_contamination_filter(index, 13)
    memory = make_exp020_contamination_filter(index, 13, index_mode="memory")
    text = "prefix one two three four five six seven eight nine ten eleven twelve thirteen fourteen suffix"
    assert default.screen(text).as_dict() == memory.screen(text).as_dict() == reference.screen(text).as_dict()
    default.close()
    memory.close()


def test_exp020_prepare_keeps_sqlite_index_mode_as_the_default() -> None:
    """Breaks if the full builder silently changes its established index mode."""
    from gibc_llm.exp020 import prepare_exp020

    assert inspect.signature(prepare_exp020).parameters["contamination_index_mode"].default == "sqlite"
