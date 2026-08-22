"""Deterministic FineWeb preparation and normalized 13-gram screening."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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


def _ngram_hashes(text: str, ngram_size: int) -> set[str]:
    tokens = normalize_for_ngrams(text).split()
    if len(tokens) < ngram_size:
        return set()
    return {
        hashlib.sha256("\u241f".join(tokens[index : index + ngram_size]).encode("utf-8")).hexdigest()
        for index in range(len(tokens) - ngram_size + 1)
    }


class NgramContaminationFilter:
    """Privacy-preserving normalized n-gram overlap index."""

    def __init__(self, ngram_hashes: set[str], ngram_size: int, benchmark_source_hashes: list[str] | None = None) -> None:
        self.ngram_hashes = ngram_hashes
        self.ngram_size = ngram_size
        self.benchmark_source_hashes = benchmark_source_hashes or []

    @classmethod
    def from_texts(cls, texts: Iterable[str], ngram_size: int = 13) -> "NgramContaminationFilter":
        hashes: set[str] = set()
        source_hashes: list[str] = []
        for text in texts:
            canonical = normalize_for_ngrams(text)
            source_hashes.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
            hashes.update(_ngram_hashes(text, ngram_size))
        return cls(hashes, ngram_size, source_hashes)

    def screen(self, text: str) -> ContaminationDecision:
        overlap = sorted(_ngram_hashes(text, self.ngram_size) & self.ngram_hashes)
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


def tensor_sha256(values: torch.Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resolve_fineweb_revision(data_config: DataConfig) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(data_config.dataset_repo, revision=data_config.dataset_revision)
    if not info.sha:
        raise RuntimeError("Hugging Face did not return an immutable FineWeb dataset revision.")
    return info.sha


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
        with archive.open("physicaliqa-train-dev/dev.jsonl") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                yield _row_to_benchmark_text("piqa", row)
    test_response = requests.get("https://yonatanbisk.com/piqa/data/tests.jsonl", timeout=120)
    test_response.raise_for_status()
    for raw_line in test_response.text.splitlines():
        yield _row_to_benchmark_text("piqa", json.loads(raw_line))


def build_benchmark_filter(cache_dir: Path, ngram_size: int) -> tuple[NgramContaminationFilter, list[dict[str, str]]]:
    """Build a local-only index; returned metadata deliberately excludes benchmark contents."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sources = [
        ("hellaswag", "hellaswag", None, ["validation"]),
        ("arc_easy", "ai2_arc", "ARC-Easy", ["validation", "test"]),
        ("winogrande", "winogrande", "winogrande_xl", ["validation"]),
        ("wikitext103", "wikitext", "wikitext-103-raw-v1", ["validation", "test"]),
    ]
    texts: list[str] = []
    metadata: list[dict[str, str]] = []
    api = HfApi()
    piqa_info = api.dataset_info("piqa")
    metadata.append(
        {
            "task": "piqa",
            "repo": "piqa",
            "config": "plain_text",
            "revision": piqa_info.sha or "unknown",
            "loader": "official_static_urls_without_remote_code",
        }
    )
    texts.extend(_stream_piqa_texts(cache_dir / "piqa"))
    for task, repo, config, splits in sources:
        info = api.dataset_info(repo)
        metadata.append(
            {"task": task, "repo": repo, "config": config or "default", "revision": info.sha or "unknown", "loader": "datasets_streaming"}
        )
        for split in splits:
            dataset = load_dataset(
                repo, name=config, split=split, revision=info.sha, cache_dir=str(cache_dir), streaming=True
            )
            texts.extend(_row_to_benchmark_text(task, row) for row in dataset)
    return NgramContaminationFilter.from_texts(texts, ngram_size=ngram_size), metadata


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


def prepare_exp001(config: ExperimentConfig, artifact_dir: Path) -> dict[str, Any]:
    """Materialize only bounded, ignored EXP-001A token tensors and provenance metadata."""
    from .tokenizer import train_tokenizer

    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifact_dir / "cache"
    revision = resolve_fineweb_revision(config.data)
    contamination_filter, benchmark_sources = build_benchmark_filter(cache_dir / "benchmarks", config.data.contamination_ngram_size)
    training_documents: list[Document] = []
    validation_documents: list[Document] = []
    scanned = accepted = rejected = 0
    train_bytes = validation_bytes = 0
    validation_byte_floor = max(2 * 1024 * 1024, config.training.smoke_validation_tokens * 8)
    for text in iter_fineweb_documents(config.data, revision, cache_dir / "fineweb"):
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
        if split == "train" and train_bytes < config.data.tokenizer_training_text_bytes:
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
    held_out_sample = validation_documents[0].text[:100_000]
    held_out_ids = tokenizer_artifact.tokenizer.encode(held_out_sample).ids
    words = re.findall(r"\S+", held_out_sample)
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
