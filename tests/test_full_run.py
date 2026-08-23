import json
from pathlib import Path

import pytest
import torch

from gibc_llm.data import tensor_sha256
from gibc_llm.full_run import dry_run_plan, expected_run_state, load_full_run_artifact
from gibc_llm.utils import load_config


def _full_artifact(tmp_path: Path) -> Path:
    config = load_config(Path("configs/exp001.yaml"))
    artifact = tmp_path / "full"
    artifact.mkdir()
    tokenizer_dir = artifact / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_path.write_text("fixture frozen tokenizer", encoding="utf-8")
    # A sparse fixture verifies the exact metadata/size contract without storing a real corpus in tests.
    stored = config.training.full_training_tokens + 1
    stream_path = artifact / "train-token-stream.uint16"
    with stream_path.open("wb") as handle:
        handle.truncate(stored * 2)
    validation_inputs = torch.arange(131_072, dtype=torch.long).reshape(256, 512) % 8192
    validation_targets = (validation_inputs + 1) % 8192
    torch.save({"inputs": validation_inputs, "targets": validation_targets}, artifact / "validation.pt")
    manifest = {
        "experiment_id": "EXP-001",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "tokenizer": {"vocab_size": 8192, "special_tokens": ["<|endoftext|>"], "sha256": "tokenizer-fixture-hash"},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": 100_007_936, "train_token_count_including_final_target": stored, "train_examples": 195_328, "train_stream_file": stream_path.name, "train_stream_bytes": stream_path.stat().st_size, "train_stream_sha256": "stream-fixture-hash", "non_cycled": True},
        "validation": {"file": "validation.pt", "prediction_tokens": 131_072, "inputs_sha256": tensor_sha256(validation_inputs), "targets_sha256": tensor_sha256(validation_targets)},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


def test_full_manifest_requires_self_contained_stream_tokenizer_and_validation(tmp_path: Path, monkeypatch) -> None:
    """Breaks if the full runner can accept a partial or mismatched full-data artifact."""
    config = load_config(Path("configs/exp001.yaml"))
    artifact = _full_artifact(tmp_path)
    def fixture_hash(path: Path) -> str:
        return "tokenizer-fixture-hash" if Path(path).name == "tokenizer.json" else "stream-fixture-hash"
    monkeypatch.setattr("gibc_llm.full_run.sha256_file", fixture_hash)
    loaded = load_full_run_artifact(artifact, config)

    assert len(loaded.train) == 195_328
    assert loaded.validation_targets.numel() == 131_072
    (artifact / "validation.pt").unlink()
    with pytest.raises(RuntimeError, match="validation"):
        load_full_run_artifact(artifact, config)


def test_full_run_arithmetic_and_explicit_dry_run_status() -> None:
    """Breaks if a bounded invocation changes the full 3052-step horizon or cursor math."""
    config = load_config(Path("configs/exp001.yaml"))
    requested, incomplete = dry_run_plan(config, start_step=0, max_steps=5)
    assert (requested, incomplete) == (5, True)
    assert expected_run_state(config, 0, requested) == (5, 163_840, 320)
    assert expected_run_state(config, 5, 1) == (6, 196_608, 384)
    assert dry_run_plan(config, start_step=0, max_steps=None) == (3052, False)
