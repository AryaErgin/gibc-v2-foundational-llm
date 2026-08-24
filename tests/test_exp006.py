"""EXP-006 preparation controls: 3x Data Recipe v1, prefix gate, and runner arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gibc_llm.data import Document, stable_document_id
from gibc_llm.exp004 import GlobalDeduplicatedTokenMixer
from gibc_llm.exp006 import EXP004_PREFIX_BYTE_COUNT, EXP004_PREFIX_SHA256, verify_stream_prefix
from gibc_llm.full_run import (
    assert_physical_batch_control,
    dry_run_plan,
    expected_full_sequences,
    expected_run_state,
    full_run_milestones,
    load_full_run_artifact,
)
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _Tokenizer:
    def encode(self, text: str) -> _Encoded:
        source, index = text.split("-")
        base = 100 if source == "fineweb" else 200
        return _Encoded([base + int(index) * 3, base + int(index) * 3 + 1, base + int(index) * 3 + 2])


def _documents(source: str, count: int = 20) -> list[Document]:
    return [Document(stable_document_id(f"{source}-{index}"), f"{source}-{index}", "train") for index in range(count)]


def _mixer(targets: dict[str, int]) -> GlobalDeduplicatedTokenMixer:
    return GlobalDeduplicatedTokenMixer(
        {"fineweb": iter(_documents("fineweb")), "fineweb_edu": iter(_documents("fineweb_edu"))},
        _Tokenizer(),
        eod_id=99,
        target_prediction_tokens=targets,
        stored_token_count=sum(targets.values()) + 1,
    )


def test_exp006_config_keeps_exp005b_model_controls_and_declares_exact_3x_recipe() -> None:
    """Breaks if EXP-006 changes an accepted architecture/training control or misses exact 3x arithmetic."""
    exp005b = load_config(Path("configs/exp005b.yaml"))
    exp006 = load_config(Path("configs/exp006.yaml"))

    assert exp006.experiment_id == "EXP-006"
    assert exp006.model == exp005b.model
    assert parameter_breakdown(DecoderOnlyTransformer(exp006.model)).total == 20_848_512
    assert exp006.training.default_microbatch_sequences == 32
    assert exp006.training.default_gradient_accumulation_steps == 2
    assert exp006.training.full_schedule_steps == 27_468
    assert exp006.training.full_training_tokens == 900_071_424
    assert exp006.training.full_schedule_steps * exp006.training.effective_batch_tokens == exp006.training.full_training_tokens
    assert expected_full_sequences(exp006) == 1_757_952
    assert exp006.mixture is not None
    assert exp006.mixture["target_prediction_tokens"] == {"fineweb": 600_047_616, "fineweb_edu": 300_023_808}
    assert sum(exp006.mixture["target_prediction_tokens"].values()) == exp006.training.full_training_tokens
    assert exp006.mixture["global_deduplication"] == "canonical_content_sha256"
    assert_physical_batch_control(exp006, 32, 2)
    with pytest.raises(RuntimeError, match="physical batch"):
        assert_physical_batch_control(exp006, 16, 4)


def test_exp006_threefold_token_deficit_targets_preserve_the_exp004_fixture_prefix() -> None:
    """Breaks if target scaling changes source selection/order before the complete EXP-004 stream prefix."""
    exp004_stream = list(_mixer({"fineweb": 8, "fineweb_edu": 4}))
    exp006_stream = list(_mixer({"fineweb": 24, "fineweb_edu": 12}))

    assert exp006_stream[: len(exp004_stream)] == exp004_stream


def test_prefix_verifier_hashes_exact_raw_bytes_and_hard_fails_mismatches(tmp_path: Path) -> None:
    """Breaks if a wrong raw prefix can receive a training-authorized manifest."""
    stream = tmp_path / "stream.uint16"
    stream.write_bytes(b"abcdefghi")

    expected = "cb5e1c4f4d6c1b4d32e1c0c4fb71f9a17b44c0f4a5d2f9a730530c7b16d9a3e4"
    with pytest.raises(RuntimeError, match="prefix SHA-256 mismatch"):
        verify_stream_prefix(stream, byte_count=6, expected_sha256=expected)

    import hashlib

    observed = hashlib.sha256(b"abcdef").hexdigest()
    assert verify_stream_prefix(stream, byte_count=6, expected_sha256=observed) == observed
    assert EXP004_PREFIX_BYTE_COUNT == 600_047_618
    assert EXP004_PREFIX_SHA256 == "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1"


def test_exp006_milestones_dry_run_and_resume_cursor_use_the_full_27468_step_horizon() -> None:
    """Breaks if a bounded invocation compresses the 900M schedule or loses the sequential cursor."""
    config = load_config(Path("configs/exp006.yaml"))

    assert full_run_milestones(config) == (0, 9_156, 18_312, 27_468)
    assert dry_run_plan(config, 0, 60) == (60, True)
    assert dry_run_plan(config, 60, 1) == (1, True)
    assert dry_run_plan(config, 0, None) == (27_468, False)
    assert expected_run_state(config, 0, 60) == (60, 1_966_080, 3_840)
    assert expected_run_state(config, 60, 1) == (61, 1_998_848, 3_904)


def test_exp006_artifact_validation_rechecks_the_manifest_prefix_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Breaks if a manifest can mark an unverified/mismatched EXP-004 prefix as training-ready."""
    config = load_config(Path("configs/exp006.yaml"))
    artifact = tmp_path / "exp006"
    (artifact / "tokenizer").mkdir(parents=True)
    (artifact / "tokenizer" / "tokenizer.json").write_text("fixture", encoding="utf-8")
    stream = artifact / "train-token-stream.uint16"
    with stream.open("wb") as handle:
        handle.truncate((config.training.full_training_tokens + 1) * 2)
    tensors = [
        ("general_validation.pt", torch.zeros((256, 512), dtype=torch.long), torch.ones((256, 512), dtype=torch.long)),
        ("edu_validation.pt", torch.full((256, 512), 2, dtype=torch.long), torch.full((256, 512), 3, dtype=torch.long)),
    ]
    for name, inputs, targets in tensors:
        torch.save({"inputs": inputs, "targets": targets}, artifact / name)

    manifest = {
        "experiment_id": "EXP-006",
        "preparation_mode": "full_stream",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "tokenizer": {"sha256": "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14", "vocab_size": 8192, "special_tokens": ["<|endoftext|>"]},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": 900_071_424, "train_token_count_including_final_target": 900_071_425, "train_examples": 1_757_952, "train_stream_file": stream.name, "train_stream_bytes": stream.stat().st_size, "train_stream_sha256": "full-stream-hash", "non_cycled": True},
        "general_validation": {"file": "general_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7", "targets_sha256": "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"},
        "edu_validation": {"file": "edu_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a", "targets_sha256": "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4", "contamination_screened": True},
        "mixture": {**config.mixture, "actual_prediction_token_contributions": {"fineweb": 600_047_616, "fineweb_edu": 300_023_808}, "unique_document_count": 1},
        "exp004_prefix": {"byte_count": EXP004_PREFIX_BYTE_COUNT, "expected_sha256": EXP004_PREFIX_SHA256, "observed_sha256": EXP004_PREFIX_SHA256, "prefix_match": True},
    }
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "gibc_llm.full_run.sha256_file",
        lambda path: "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14" if Path(path).name == "tokenizer.json" else "full-stream-hash",
    )
    monkeypatch.setattr("gibc_llm.full_run.sha256_file_prefix", lambda *_: EXP004_PREFIX_SHA256)
    monkeypatch.setattr(
        "gibc_llm.full_run.tensor_sha256",
        lambda values: "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7" if int(values[0, 0]) == 0 else "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e" if int(values[0, 0]) == 1 else "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a" if int(values[0, 0]) == 2 else "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4",
    )
    assert load_full_run_artifact(artifact, config).manifest["experiment_id"] == "EXP-006"

    manifest["exp004_prefix"]["prefix_match"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="verified exact EXP-004 byte prefix"):
        load_full_run_artifact(artifact, config)
