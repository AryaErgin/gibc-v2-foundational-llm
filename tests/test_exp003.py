import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from gibc_llm.data import tensor_sha256
from gibc_llm.full_run import (
    assert_physical_batch_control,
    dry_run_plan,
    expected_full_sequences,
    expected_run_state,
    load_full_run_artifact,
)
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp003_controls_match_exp002_except_for_data_source() -> None:
    """Breaks if the FineWeb-Edu ablation drifts any fixed EXP-002 training control."""
    exp002 = load_config(Path("configs/exp002.yaml"))
    exp003 = load_config(Path("configs/exp003.yaml"))

    assert exp003.experiment_id == "EXP-003"
    assert exp003.model == exp002.model
    assert exp003.training == exp002.training
    assert exp003.data.dataset_repo == "HuggingFaceFW/fineweb-edu"
    assert exp003.data.dataset_config == "default"
    assert exp003.data.dataset_revision == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    for field, value in asdict(exp002.data).items():
        if field not in {"dataset_repo", "dataset_config", "dataset_revision"}:
            assert asdict(exp003.data)[field] == value
    assert exp003.training.default_microbatch_sequences == 32
    assert exp003.training.default_gradient_accumulation_steps == 2
    assert exp003.training.full_schedule_steps * exp003.training.effective_batch_tokens == 300_023_808
    assert expected_full_sequences(exp003) == 585_984


def test_exp003_fixed_model_and_full_path_dry_run_arithmetic() -> None:
    """Breaks if EXP-003 changes model size or lets an explicit dry run rewrite its 9,156-step horizon."""
    config = load_config(Path("configs/exp003.yaml"))
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 8_392_960
    assert dry_run_plan(config, start_step=0, max_steps=5) == (5, True)
    assert expected_run_state(config, 0, 5) == (5, 163_840, 320)
    assert dry_run_plan(config, start_step=0, max_steps=None) == (9156, False)
    assert_physical_batch_control(config, microbatch_sequences=32, accumulation_steps=2)
    with pytest.raises(RuntimeError, match="physical batch"):
        assert_physical_batch_control(config, microbatch_sequences=16, accumulation_steps=4)


def test_exp003_frozen_artifacts_reject_wrong_tokenizer_or_general_validation(tmp_path: Path, monkeypatch) -> None:
    """Breaks if EXP-003 can use a retrained tokenizer or altered generic validation control."""
    from gibc_llm.exp003 import assert_frozen_exp003_artifacts

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("not the approved frozen tokenizer", encoding="utf-8")
    validation = tmp_path / "validation.pt"
    torch.save({"inputs": torch.zeros((256, 512), dtype=torch.long), "targets": torch.ones((256, 512), dtype=torch.long)}, validation)

    with pytest.raises(RuntimeError, match="frozen EXP-001 tokenizer"):
        assert_frozen_exp003_artifacts(tokenizer, validation)

    monkeypatch.setattr("gibc_llm.exp003.sha256_file", lambda _: "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14")
    with pytest.raises(RuntimeError, match="general validation"):
        assert_frozen_exp003_artifacts(tokenizer, validation)


def _exp003_artifact(tmp_path: Path) -> Path:
    config = load_config(Path("configs/exp003.yaml"))
    artifact = tmp_path / "exp003"
    (artifact / "tokenizer").mkdir(parents=True)
    (artifact / "tokenizer" / "tokenizer.json").write_text("fixture", encoding="utf-8")
    stored = config.training.full_training_tokens + 1
    stream = artifact / "train-token-stream.uint16"
    with stream.open("wb") as handle:
        handle.truncate(stored * 2)
    general_inputs = torch.zeros((256, 512), dtype=torch.long)
    general_targets = torch.ones((256, 512), dtype=torch.long)
    edu_inputs = torch.full((256, 512), 2, dtype=torch.long)
    edu_targets = torch.full((256, 512), 3, dtype=torch.long)
    torch.save({"inputs": general_inputs, "targets": general_targets}, artifact / "general_validation.pt")
    torch.save({"inputs": edu_inputs, "targets": edu_targets}, artifact / "edu_validation.pt")
    manifest = {
        "experiment_id": "EXP-003",
        "preparation_mode": "full_stream",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field, "license": "ODC-BY", "provenance": "filtered public FineWeb"},
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14", "vocab_size": 8192, "special_tokens": ["<|endoftext|>"]},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": 300_023_808, "train_token_count_including_final_target": stored, "train_examples": 585_984, "train_stream_file": stream.name, "train_stream_bytes": stream.stat().st_size, "train_stream_sha256": "stream-hash", "non_cycled": True},
        "general_validation": {"file": "general_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7", "targets_sha256": "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"},
        "edu_validation": {"file": "edu_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": tensor_sha256(edu_inputs), "targets_sha256": tensor_sha256(edu_targets), "split": {"method": "sha256(seed:canonical_content_sha256) modulo buckets", "seed": 42, "validation_buckets": 200, "modulus": 10000}, "contamination_screened": True},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


def test_exp003_full_manifest_requires_both_frozen_validations_and_no_cycling(tmp_path: Path, monkeypatch) -> None:
    """Breaks if the EXP-003 runner accepts an artifact without two provenance-checked held-out validations."""
    config = load_config(Path("configs/exp003.yaml"))
    artifact = _exp003_artifact(tmp_path)

    def fixture_file_hash(path: Path) -> str:
        return "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14" if Path(path).name == "tokenizer.json" else "stream-hash"

    def fixture_tensor_hash(values: torch.Tensor) -> str:
        if int(values[0, 0]) == 0:
            return "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7"
        if int(values[0, 0]) == 1:
            return "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"
        return tensor_sha256(values)

    monkeypatch.setattr("gibc_llm.full_run.sha256_file", fixture_file_hash)
    monkeypatch.setattr("gibc_llm.full_run.tensor_sha256", fixture_tensor_hash)
    loaded = load_full_run_artifact(artifact, config)
    assert loaded.edu_validation_targets is not None
    assert loaded.edu_validation_targets.numel() == 131_072
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["packed"]["non_cycled"] = False
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-cycling"):
        load_full_run_artifact(artifact, config)

    manifest["packed"]["non_cycled"] = True
    manifest["preparation_mode"] = "validation_only"
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete stream"):
        load_full_run_artifact(artifact, config)


def test_exp003_full_runner_rejects_a_non_frozen_tokenizer_hash(tmp_path: Path, monkeypatch) -> None:
    """Breaks if a self-consistent but retrained tokenizer can reach the EXP-003 training runner."""
    config = load_config(Path("configs/exp003.yaml"))
    artifact = _exp003_artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["sha256"] = "not-the-approved-tokenizer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "gibc_llm.full_run.sha256_file",
        lambda path: "not-the-approved-tokenizer" if Path(path).name == "tokenizer.json" else "stream-hash",
    )
    monkeypatch.setattr(
        "gibc_llm.full_run.tensor_sha256",
        lambda values: "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7"
        if int(values[0, 0]) == 0
        else "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"
        if int(values[0, 0]) == 1
        else tensor_sha256(values),
    )
    with pytest.raises(RuntimeError, match="frozen tokenizer"):
        load_full_run_artifact(artifact, config)


def test_exp003_milestones_are_the_predeclared_scaling_curve() -> None:
    """Breaks if EXP-003 drops or moves a required two-validation milestone."""
    from gibc_llm.exp003 import scaling_milestones

    config = load_config(Path("configs/exp003.yaml"))
    assert scaling_milestones(config) == (0, 3052, 6104, 9156)
