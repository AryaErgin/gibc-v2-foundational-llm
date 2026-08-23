import json
from pathlib import Path

import pytest
import torch

from gibc_llm.data import Document, NgramContaminationFilter, stable_document_id, tensor_sha256
from gibc_llm.full_run import assert_physical_batch_control, dry_run_plan, expected_full_sequences, expected_run_state, load_full_run_artifact
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _Tokenizer:
    def encode(self, text: str) -> _Encoded:
        return _Encoded({"shared": [11, 12, 13], "generic": [21, 22, 23], "generic_2": [24, 25, 26], "educational": [31, 32, 33]}[text])


def _document(text: str) -> Document:
    return Document(document_id=stable_document_id(text), text=text, split="train")


def test_exp004_config_keeps_exp002_controls_and_declares_only_mixture_data() -> None:
    """Breaks if EXP-004 changes a fixed model/training control instead of only data composition."""
    exp002 = load_config(Path("configs/exp002.yaml"))
    exp004 = load_config(Path("configs/exp004.yaml"))

    assert exp004.experiment_id == "EXP-004"
    assert exp004.model == exp002.model
    assert exp004.training == exp002.training
    assert exp004.data == exp002.data
    assert exp004.mixture is not None
    assert exp004.mixture["target_prediction_tokens"] == {"fineweb": 200_015_872, "fineweb_edu": 100_007_936}
    assert exp004.mixture["global_deduplication"] == "canonical_content_sha256"
    assert expected_full_sequences(exp004) == 585_984


def test_global_deduplicated_mixture_is_deterministic_and_records_source_token_contributions() -> None:
    """Breaks if overlapping FineWeb-Edu content is duplicated or source-token composition is untracked."""
    from gibc_llm.exp004 import GlobalDeduplicatedTokenMixer

    sources = {
        "fineweb": iter([_document("shared"), _document("generic"), _document("generic_2")]),
        "fineweb_edu": iter([_document("shared"), _document("educational")]),
    }
    mixer = GlobalDeduplicatedTokenMixer(
        sources,
        _Tokenizer(),
        eod_id=99,
        target_prediction_tokens={"fineweb": 8, "fineweb_edu": 4},
        stored_token_count=13,
    )
    first = list(mixer)

    repeat = GlobalDeduplicatedTokenMixer(
        {
            "fineweb": iter([_document("shared"), _document("generic"), _document("generic_2")]),
            "fineweb_edu": iter([_document("shared"), _document("educational")]),
        },
        _Tokenizer(),
        eod_id=99,
        target_prediction_tokens={"fineweb": 8, "fineweb_edu": 4},
        stored_token_count=13,
    )
    assert first == list(repeat)
    assert len(first) == 13
    assert mixer.prediction_token_contributions == {"fineweb": 8, "fineweb_edu": 4}
    assert mixer.cross_source_duplicates_skipped == {"fineweb": 0, "fineweb_edu": 1}
    assert mixer.selected_document_ids == {stable_document_id("shared"), stable_document_id("generic"), stable_document_id("generic_2"), stable_document_id("educational")}


def test_exp004_frozen_controls_and_full_path_arithmetic() -> None:
    """Breaks if EXP-004 accepts changed validation controls or a different full-run shape/batch."""
    from gibc_llm.exp004 import assert_frozen_exp004_artifacts

    config = load_config(Path("configs/exp004.yaml"))
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 8_392_960
    assert dry_run_plan(config, 0, 5) == (5, True)
    assert expected_run_state(config, 0, 5) == (5, 163_840, 320)
    assert_physical_batch_control(config, 32, 2)
    with pytest.raises(RuntimeError, match="physical batch"):
        assert_physical_batch_control(config, 16, 4)

    tokenizer = Path("__missing_tokenizer__")
    with pytest.raises(RuntimeError, match="frozen"):
        assert_frozen_exp004_artifacts(tokenizer, Path("__missing_general__"), Path("__missing_edu__"))


def test_exp004_source_iterator_screens_contamination_before_global_deduplication(monkeypatch, tmp_path: Path) -> None:
    """Breaks if a contaminated source document can reach the mixture selector merely because it is unique."""
    from gibc_llm.exp004 import _screened_train_documents

    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    contaminated = f"prefix {benchmark} suffix"
    clean = "unique material about lanterns, river gauges, and deterministically sampled documents"
    counters = {"scanned_documents": 0, "accepted_documents": 0, "rejected_documents": 0, "validation_documents_excluded": 0}
    config = load_config(Path("configs/exp004.yaml")).data
    monkeypatch.setattr("gibc_llm.exp004.iter_fineweb_documents", lambda *_: iter([contaminated, clean]))
    monkeypatch.setattr("gibc_llm.exp004.assign_split", lambda *_: "train")

    documents = list(_screened_train_documents(config, tmp_path, NgramContaminationFilter.from_texts([benchmark]), counters))
    assert [document.text for document in documents] == [clean]
    assert counters == {"scanned_documents": 2, "accepted_documents": 1, "rejected_documents": 1, "validation_documents_excluded": 0}


def test_exp004_full_manifest_requires_frozen_two_validation_controls_and_complete_source_accounting(tmp_path: Path, monkeypatch) -> None:
    """Breaks if the runner accepts altered validation tensors or a mixture without complete source-token accounting."""
    config = load_config(Path("configs/exp004.yaml"))
    artifact = tmp_path / "exp004"
    (artifact / "tokenizer").mkdir(parents=True)
    (artifact / "tokenizer" / "tokenizer.json").write_text("fixture", encoding="utf-8")
    stream = artifact / "train-token-stream.uint16"
    stored = config.training.full_training_tokens + 1
    with stream.open("wb") as handle:
        handle.truncate(stored * 2)
    general_inputs = torch.zeros((256, 512), dtype=torch.long)
    general_targets = torch.ones((256, 512), dtype=torch.long)
    edu_inputs = torch.full((256, 512), 2, dtype=torch.long)
    edu_targets = torch.full((256, 512), 3, dtype=torch.long)
    torch.save({"inputs": general_inputs, "targets": general_targets}, artifact / "general_validation.pt")
    torch.save({"inputs": edu_inputs, "targets": edu_targets}, artifact / "edu_validation.pt")
    manifest = {
        "experiment_id": "EXP-004", "preparation_mode": "full_stream",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "tokenizer": {"sha256": "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14", "vocab_size": 8192, "special_tokens": ["<|endoftext|>"]},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": 300_023_808, "train_token_count_including_final_target": stored, "train_examples": 585_984, "train_stream_file": stream.name, "train_stream_bytes": stream.stat().st_size, "train_stream_sha256": "stream-hash", "non_cycled": True},
        "general_validation": {"file": "general_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7", "targets_sha256": "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"},
        "edu_validation": {"file": "edu_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a", "targets_sha256": "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4", "contamination_screened": True},
        "mixture": {**config.mixture, "actual_prediction_token_contributions": {"fineweb": 200_015_872, "fineweb_edu": 100_007_936}, "unique_document_count": 1},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr("gibc_llm.full_run.sha256_file", lambda path: "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14" if Path(path).name == "tokenizer.json" else "stream-hash")
    monkeypatch.setattr(
        "gibc_llm.full_run.tensor_sha256",
        lambda values: "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7" if int(values[0, 0]) == 0 else "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e" if int(values[0, 0]) == 1 else "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a" if int(values[0, 0]) == 2 else "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4",
    )
    assert load_full_run_artifact(artifact, config).edu_validation_targets is not None
    manifest["mixture"]["actual_prediction_token_contributions"] = {"fineweb": 1, "fineweb_edu": 1}
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-token accounting"):
        load_full_run_artifact(artifact, config)
