"""Deterministic FineWeb preparation and normalized 13-gram screening."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sqlite3
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from .utils import DataConfig, ExperimentConfig, atomic_json_write, sha256_file


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    split: str


@dataclass(frozen=True)
class ContaminationDecision:
    rejected: bool
    document_sha256: str
    overlap_count: int
    overlap_hashes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rejected": self.rejected,
            "document_sha256": self.document_sha256,
            "overlap_count": self.overlap_count,
            "overlap_hashes": list(self.overlap_hashes),
        }


def canonical_document_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def stable_document_id(text: str) -> str:
    return hashlib.sha256(canonical_document_text(text).encode("utf-8")).hexdigest()


def assign_split(document_id: str, seed: int, modulus: int, validation_cutoff: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).hexdigest(), 16) % modulus
    return "validation" if bucket < validation_cutoff else "train"


def normalize_for_ngrams(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE))


def _ngram_hashes(text: str, ngram_size: int) -> set[bytes]:
    tokens = normalize_for_ngrams(text).split()
    if len(tokens) < ngram_size:
        return set()
    return {
        hashlib.sha256("\u241f".join(tokens[index : index + ngram_size]).encode("utf-8")).digest()
        for index in range(len(tokens) - ngram_size + 1)
    }


class NgramContaminationFilter:
    """Privacy-preserving normalized n-gram overlap index."""

    def __init__(
        self,
        ngram_hashes: set[bytes] | None,
        ngram_size: int,
        benchmark_source_hashes: list[str] | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        self.ngram_hashes = ngram_hashes
        self.ngram_size = ngram_size
        self.benchmark_source_hashes = benchmark_source_hashes or []
        self.sqlite_path = sqlite_path

    @classmethod
    def from_texts(cls, texts: Iterable[str], ngram_size: int = 13) -> "NgramContaminationFilter":
        hashes: set[bytes] = set()
        source_hashes: list[str] = []
        for text in texts:
            canonical = normalize_for_ngrams(text)
            source_hashes.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
            hashes.update(_ngram_hashes(text, ngram_size))
        return cls(hashes, ngram_size, source_hashes)

    @classmethod
    def from_sqlite_texts(cls, texts: Iterable[str], path: Path, ngram_size: int = 13) -> "NgramContaminationFilter":
        """Build a hash-only on-disk index without retaining benchmark text in RAM."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as connection:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("CREATE TABLE IF NOT EXISTS ngram_hashes (value BLOB PRIMARY KEY)")
            pending: list[tuple[bytes]] = []
            for text in texts:
                pending.extend((value,) for value in _ngram_hashes(text, ngram_size))
                if len(pending) >= 20_000:
                    connection.executemany("INSERT OR IGNORE INTO ngram_hashes(value) VALUES (?)", pending)
                    pending.clear()
            if pending:
                connection.executemany("INSERT OR IGNORE INTO ngram_hashes(value) VALUES (?)", pending)
        return cls(None, ngram_size, sqlite_path=target)

    def screen(self, text: str) -> ContaminationDecision:
        candidates = _ngram_hashes(text, self.ngram_size)
        if self.ngram_hashes is not None:
            overlap = sorted(value.hex() for value in (candidates & self.ngram_hashes))
        elif self.sqlite_path is not None:
            matched: set[bytes] = set()
            with sqlite3.connect(self.sqlite_path) as connection:
                candidate_list = list(candidates)
                for start in range(0, len(candidate_list), 900):
                    chunk = candidate_list[start : start + 900]
                    if chunk:
                        placeholders = ",".join("?" for _ in chunk)
                        matched.update(row[0] for row in connection.execute(f"SELECT value FROM ngram_hashes WHERE value IN ({placeholders})", chunk))
            overlap = sorted(value.hex() for value in matched)
        else:
            raise RuntimeError("Contamination filter has neither in-memory nor on-disk index.")
        return ContaminationDecision(
            rejected=bool(overlap),
            document_sha256=stable_document_id(text),
            overlap_count=len(overlap),
            overlap_hashes=tuple(overlap),
        )


def pack_documents(documents: Iterable[Sequence[int]], eod_id: int) -> torch.Tensor:
    stream: list[int] = []
    for token_ids in documents:
        stream.extend(token_ids)
        stream.append(eod_id)
    return torch.tensor(stream, dtype=torch.long)


def make_packed_examples(stream: torch.Tensor, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if stream.ndim != 1:
        raise ValueError("Packed token stream must be one-dimensional.")
    starts = range(0, stream.numel() - context_length, context_length)
    inputs = [stream[start : start + context_length] for start in starts]
    targets = [stream[start + 1 : start + context_length + 1] for start in starts]
    if not inputs:
        raise ValueError("Packed stream does not contain one 512-prediction example.")
    return torch.stack(inputs), torch.stack(targets)


class TokenStreamDataset:
    """Read-only 1-D uint16 token storage exposing 513-token shifted examples.

    The representation stores every token once.  It deliberately creates
    ``torch.long`` inputs only for the requested training microbatch, immediately
    before embedding lookup, rather than materializing duplicate input/target
    tensors for a full training corpus.
    """

    storage_dtype = "uint16"

    def __init__(self, path: Path, token_count: int, context_length: int = 512) -> None:
        if token_count < context_length + 1:
            raise ValueError("Token stream does not contain one complete next-token example.")
        self.path = Path(path)
        self.token_count = token_count
        self.context_length = context_length
        self._token_ids: np.memmap | None = None

    def __len__(self) -> int:
        return (self.token_count - 1) // self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.context_length
        token_ids = self._open()
        # Copy avoids retaining a read-only memmap warning while safely promoting
        # exactly the requested 513 tokens for PyTorch's embedding input.
        window = np.array(token_ids[start : start + self.context_length + 1], dtype=np.int64, copy=True)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])

    def _open(self) -> np.memmap:
        """Open the immutable stream once per dataset instance (including on Windows)."""
        if self._token_ids is None:
            self._token_ids = np.memmap(self.path, mode="r", dtype=np.uint16, shape=(self.token_count,))
        return self._token_ids

    def get_contiguous_batch(self, start_sequence: int, sequence_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return consecutive shifted examples from one contiguous underlying slice."""
        if sequence_count <= 0 or start_sequence < 0 or start_sequence + sequence_count > len(self):
            raise IndexError((start_sequence, sequence_count))
        start = start_sequence * self.context_length
        stop = (start_sequence + sequence_count) * self.context_length + 1
        window = np.array(self._open()[start:stop], dtype=np.int64, copy=True)
        inputs = torch.from_numpy(window[:-1].reshape(sequence_count, self.context_length))
        targets = torch.from_numpy(window[1:].reshape(sequence_count, self.context_length))
        return inputs, targets


def write_token_stream(
    path: Path, token_ids: Iterable[int], token_count: int, context_length: int = 512
) -> TokenStreamDataset:
    """Write one exact uint16 stream and reject short/overflowing token feeds."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.memmap(target, mode="w+", dtype=np.uint16, shape=(token_count,))
    written = 0
    try:
        for token_id in token_ids:
            if written >= token_count:
                raise ValueError("Token stream source exceeds the declared exact token count.")
            if not 0 <= int(token_id) < 8192:
                raise ValueError("EXP-001 uint16 token stream received an out-of-vocabulary token ID.")
            values[written] = token_id
            written += 1
        if written != token_count:
            raise ValueError(f"Token stream source ended at {written} tokens; expected {token_count}.")
        values.flush()
    finally:
        del values
    return TokenStreamDataset(target, token_count, context_length)


def tensor_sha256(values: torch.Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resolve_fineweb_revision(data_config: DataConfig) -> str:
    """Use the approved immutable SHA directly; never resolve a moving dataset HEAD."""
    if not data_config.dataset_revision:
        raise RuntimeError("EXP-001 requires an explicit immutable FineWeb dataset revision.")
    return data_config.dataset_revision


def iter_fineweb_documents(data_config: DataConfig, revision: str, cache_dir: Path) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        data_config.dataset_repo,
        name=data_config.dataset_config,
        split="train",
        streaming=True,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    for row in dataset:
        value = row.get(data_config.text_field)
        if isinstance(value, str) and value.strip():
            yield value


def _row_to_benchmark_text(task: str, row: dict[str, Any]) -> str:
    if task == "hellaswag":
        return " ".join(str(row.get(key, "")) for key in ("ctx_a", "ctx_b", "activity_label", "endings"))
    if task == "arc_easy":
        choices = row.get("choices", {})
        choice_text = " ".join(str(item) for item in choices.get("text", [])) if isinstance(choices, dict) else ""
        return f"{row.get('question', '')} {choice_text}"
    if task == "piqa":
        return " ".join(str(row.get(key, "")) for key in ("goal", "sol1", "sol2"))
    if task == "winogrande":
        return " ".join(str(row.get(key, "")) for key in ("sentence", "option1", "option2"))
    if task == "wikitext103":
        return str(row.get("text", ""))
    raise ValueError(f"Unsupported benchmark task: {task}")


def _stream_piqa_texts(cache_dir: Path) -> Iterator[str]:
    """Read PIQA's documented static files without executing its dataset loader."""
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "physicaliqa-train-dev.zip"
    if not archive_path.exists():
        response = requests.get(
            "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip", timeout=120
        )
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    with zipfile.ZipFile(archive_path) as archive:
        for member in ("physicaliqa-train-dev/train.jsonl", "physicaliqa-train-dev/dev.jsonl"):
            with archive.open(member) as handle:
                for raw_line in handle:
                    row = json.loads(raw_line)
                    yield _row_to_benchmark_text("piqa", row)
    test_response = requests.get("https://yonatanbisk.com/piqa/data/tests.jsonl", timeout=120)
    test_response.raise_for_status()
    for raw_line in test_response.text.splitlines():
        yield _row_to_benchmark_text("piqa", json.loads(raw_line))


def benchmark_sources_for_index(lock_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the committed, immutable benchmark revisions and public splits."""
    if lock_path is None:
        lock_path = Path(__file__).resolve().parents[2] / "provenance" / "exp001-benchmark-revisions.json"
    raw = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    required = {"hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"}
    if set(raw) != required:
        raise ValueError("EXP-001 benchmark provenance lock is incomplete.")
    sources = []
    for task in ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"):
        source = dict(raw[task])
        source["task"] = task
        if not isinstance(source.get("revision"), str) or len(source["revision"]) != 40:
            raise ValueError(f"Benchmark revision for {task} is not an immutable Git SHA.")
        sources.append(source)
    return sources


def build_benchmark_filter(cache_dir: Path, ngram_size: int) -> tuple[NgramContaminationFilter, list[dict[str, Any]]]:
    """Build a local-only index from every public MC split and WikiText held-out text."""
    from datasets import load_dataset

    metadata: list[dict[str, Any]] = []
    sources = benchmark_sources_for_index()
    for source in sources:
        task = source["task"]
        metadata.append(
            {
                "task": task,
                "repo": source["repo"],
                "config": source["config"] or "default",
                "revision": source["revision"],
                "splits": source["splits"],
                "loader": "official_static_urls_without_remote_code" if task == "piqa" else "datasets_streaming",
            }
        )
    def benchmark_texts() -> Iterator[str]:
        for source in sources:
            task = source["task"]
            if task == "piqa":
                yield from _stream_piqa_texts(cache_dir / "piqa")
                continue
            for split in source["splits"]:
                dataset = load_dataset(
                    source["repo"], name=source["config"], split=split, revision=source["revision"], cache_dir=str(cache_dir), streaming=True
                )
                yield from (_row_to_benchmark_text(task, row) for row in dataset)

    return NgramContaminationFilter.from_sqlite_texts(
        benchmark_texts(), cache_dir / "benchmark-ngrams.sqlite", ngram_size=ngram_size
    ), metadata


def _stream_token_ids(tokenizer: Any, documents: Iterable[Document], eod_id: int, required_tokens: int) -> tuple[torch.Tensor, int]:
    tokens: list[int] = []
    consumed = 0
    for document in documents:
        tokens.extend(tokenizer.encode(document.text).ids)
        tokens.append(eod_id)
        consumed += 1
        if len(tokens) >= required_tokens + 1:
            break
    usable = (required_tokens // 512) * 512
    return torch.tensor(tokens[: usable + 1], dtype=torch.long), consumed


def prepare_exp001(config: ExperimentConfig, artifact_dir: Path, full_run: bool = False) -> dict[str, Any]:
    """Prepare bounded EXP-001A tensors or the non-cycled full-run token stream."""
    from .tokenizer import train_tokenizer

    started = time.perf_counter()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifact_dir / "cache"
    revision = resolve_fineweb_revision(config.data)
    contamination_filter, benchmark_sources = build_benchmark_filter(cache_dir / "benchmarks", config.data.contamination_ngram_size)
    training_documents: list[Document] = []
    initial_training_documents: list[Document] = []
    validation_documents: list[Document] = []
    scanned = accepted = rejected = 0
    train_bytes = validation_bytes = 0
    validation_byte_floor = max(2 * 1024 * 1024, config.training.smoke_validation_tokens * 8)
    source_documents = iter(iter_fineweb_documents(config.data, revision, cache_dir / "fineweb"))
    for text in source_documents:
        scanned += 1
        document_id = stable_document_id(text)
        split = assign_split(
            document_id,
            config.data.split_seed,
            config.data.validation_bucket_modulus,
            config.data.validation_bucket_cutoff,
        )
        decision = contamination_filter.screen(text)
        if decision.rejected:
            rejected += 1
            continue
        accepted += 1
        document = Document(document_id=document_id, text=text, split=split)
        encoded_bytes = len(text.encode("utf-8"))
        if split == "train":
            initial_training_documents.append(document)
            if train_bytes < config.data.tokenizer_training_text_bytes:
                training_documents.append(document)
                train_bytes += encoded_bytes
        elif split == "validation" and validation_bytes < validation_byte_floor:
            validation_documents.append(document)
            validation_bytes += encoded_bytes
        if train_bytes >= config.data.tokenizer_training_text_bytes and validation_bytes >= validation_byte_floor:
            break
    if train_bytes < config.data.tokenizer_training_text_bytes or not validation_documents:
        raise RuntimeError("Bounded FineWeb stream did not produce sufficient accepted train/validation text.")
    tokenizer_artifact = train_tokenizer(
        training_documents,
        output_dir=artifact_dir / "tokenizer",
        vocab_size=config.data.tokenizer_vocab_size,
        eod_token=config.data.eod_token,
    )
    eod_id = tokenizer_artifact.tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("Serialized EXP-001 tokenizer is missing the EOD ID.")
    held_out_sample = validation_documents[0].text[:100_000]
    held_out_ids = tokenizer_artifact.tokenizer.encode(held_out_sample).ids
    words = re.findall(r"\S+", held_out_sample)
    if full_run:
        full_token_count = config.training.full_training_tokens + 1
        documents_contributed = 0

        def full_train_token_ids() -> Iterator[int]:
            nonlocal scanned, accepted, rejected, documents_contributed
            for document in initial_training_documents:
                documents_contributed += 1
                yield from tokenizer_artifact.tokenizer.encode(document.text).ids
                yield eod_id
            for text in source_documents:
                scanned += 1
                document_id = stable_document_id(text)
                split = assign_split(
                    document_id,
                    config.data.split_seed,
                    config.data.validation_bucket_modulus,
                    config.data.validation_bucket_cutoff,
                )
                decision = contamination_filter.screen(text)
                if decision.rejected:
                    rejected += 1
                    continue
                accepted += 1
                if split == "train":
                    documents_contributed += 1
                    yield from tokenizer_artifact.tokenizer.encode(text).ids
                    yield eod_id

        stream_path = artifact_dir / "train-token-stream.uint16"
        stream = write_token_stream(
            stream_path,
            itertools.islice(full_train_token_ids(), full_token_count),
            token_count=full_token_count,
            context_length=config.data.context_length,
        )
        validation_stream, validation_docs_used = _stream_token_ids(
            tokenizer_artifact.tokenizer, validation_documents, eod_id, config.training.smoke_validation_tokens
        )
        validation_inputs, validation_targets = make_packed_examples(validation_stream, config.data.context_length)
        validation_path = artifact_dir / "validation.pt"
        torch.save({"inputs": validation_inputs, "targets": validation_targets}, validation_path)
        manifest = {
            "experiment_id": config.experiment_id,
            "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": revision, "field": config.data.text_field},
            "split": {"method": "sha256(seed:canonical_content_sha256) modulo buckets", "seed": config.data.split_seed, "validation_buckets": config.data.validation_bucket_cutoff, "modulus": config.data.validation_bucket_modulus},
            "contamination": {"method": "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap", "ngram_size": config.data.contamination_ngram_size, "scanned_documents": scanned, "accepted_documents": accepted, "rejected_documents": rejected, "benchmark_sources": benchmark_sources},
            "tokenizer": {"vocab_size": tokenizer_artifact.vocab_size, "special_tokens": tokenizer_artifact.special_tokens, "sha256": tokenizer_artifact.sha256, "bytes_per_token": len(held_out_sample.encode("utf-8")) / max(1, len(held_out_ids)), "characters_per_token": len(held_out_sample) / max(1, len(held_out_ids)), "tokens_per_word": len(held_out_ids) / max(1, len(words))},
            "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": config.data.context_length, "prediction_tokens_per_example": 512, "eod_token_id": eod_id, "cross_document_loss": "permitted only across an explicit EOD token", "train_examples": len(stream), "train_prediction_tokens": config.training.full_training_tokens, "train_token_count_including_final_target": full_token_count, "train_stream_file": stream_path.name, "train_stream_bytes": stream_path.stat().st_size, "train_stream_sha256": sha256_file(stream_path), "train_documents_contributed": documents_contributed, "non_cycled": True},
            "validation": {"file": validation_path.name, "examples": int(validation_inputs.shape[0]), "prediction_tokens": int(validation_targets.numel()), "documents_used": validation_docs_used, "inputs_sha256": tensor_sha256(validation_inputs), "targets_sha256": tensor_sha256(validation_targets)},
            "preparation_wall_seconds": time.perf_counter() - started,
        }
        atomic_json_write(artifact_dir / "manifest.json", manifest)
        manifest["manifest_sha256"] = sha256_file(artifact_dir / "manifest.json")
        return manifest
    train_stream, train_docs_used = _stream_token_ids(
        tokenizer_artifact.tokenizer, training_documents, eod_id, config.training.smoke_training_tokens
    )
    validation_stream, validation_docs_used = _stream_token_ids(
        tokenizer_artifact.tokenizer, validation_documents, eod_id, config.training.smoke_validation_tokens
    )
    train_inputs, train_targets = make_packed_examples(train_stream, config.data.context_length)
    validation_inputs, validation_targets = make_packed_examples(validation_stream, config.data.context_length)
    torch.save({"inputs": train_inputs, "targets": train_targets}, artifact_dir / "train.pt")
    torch.save({"inputs": validation_inputs, "targets": validation_targets}, artifact_dir / "validation.pt")
    manifest = {
        "experiment_id": config.experiment_id,
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": revision, "field": config.data.text_field},
        "split": {"method": "sha256(seed:canonical_content_sha256) modulo buckets", "seed": config.data.split_seed, "validation_buckets": config.data.validation_bucket_cutoff, "modulus": config.data.validation_bucket_modulus},
        "contamination": {"method": "NFKC+casefold+tokenized normalized 13-gram SHA-256 overlap", "ngram_size": config.data.contamination_ngram_size, "scanned_documents": scanned, "accepted_documents": accepted, "rejected_documents": rejected, "benchmark_sources": benchmark_sources},
        "tokenizer": {"vocab_size": tokenizer_artifact.vocab_size, "special_tokens": tokenizer_artifact.special_tokens, "sha256": tokenizer_artifact.sha256, "bytes_per_token": len(held_out_sample.encode("utf-8")) / max(1, len(held_out_ids)), "characters_per_token": len(held_out_sample) / max(1, len(held_out_ids)), "tokens_per_word": len(held_out_ids) / max(1, len(words))},
        "packed": {"context_length": config.data.context_length, "prediction_tokens_per_example": 512, "eod_token_id": eod_id, "cross_document_loss": "permitted only across an explicit EOD token", "train_examples": int(train_inputs.shape[0]), "validation_examples": int(validation_inputs.shape[0]), "train_prediction_tokens": int(train_targets.numel()), "validation_prediction_tokens": int(validation_targets.numel()), "train_tensor_sha256": tensor_sha256(train_inputs), "validation_tensor_sha256": tensor_sha256(validation_inputs), "train_documents_used": train_docs_used, "validation_documents_used": validation_docs_used},
    }
    atomic_json_write(artifact_dir / "manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(artifact_dir / "manifest.json")
    return manifest
