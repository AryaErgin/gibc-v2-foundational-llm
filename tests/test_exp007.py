"""EXP-007 near-cap architecture preflight controls against the exact frozen EXP-004 artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gibc_llm.full_run import assert_physical_batch_control, dry_run_plan, expected_full_sequences, expected_run_state, full_run_milestones, load_full_run_artifact
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp007_candidates_have_exact_real_parameter_counts_and_only_allocation_changes() -> None:
    """Breaks if a near-cap allocation exceeds its declared count or changes a frozen EXP-004 control."""
    exp004 = load_config(Path("configs/exp004.yaml"))
    candidate_a = load_config(Path("configs/exp007a.yaml"))
    candidate_b = load_config(Path("configs/exp007b.yaml"))

    assert candidate_a.experiment_id == "EXP-007A"
    assert candidate_b.experiment_id == "EXP-007B"
    assert (candidate_a.model.d_model, candidate_a.model.n_layers, candidate_a.model.n_heads, candidate_a.model.head_dim, candidate_a.model.d_ff) == (608, 10, 19, 32, 2432)
    assert (candidate_b.model.d_model, candidate_b.model.n_layers, candidate_b.model.n_heads, candidate_b.model.head_dim, candidate_b.model.d_ff) == (640, 9, 20, 32, 2560)
    assert parameter_breakdown(DecoderOnlyTransformer(candidate_a.model)).total == 49_353_184
    assert parameter_breakdown(DecoderOnlyTransformer(candidate_b.model)).total == 49_491_840
    assert parameter_breakdown(DecoderOnlyTransformer(candidate_b.model)).total - parameter_breakdown(DecoderOnlyTransformer(candidate_a.model)).total == 138_656
    assert 0.0027 < 138_656 / 49_353_184 < 0.0029
    for candidate in (candidate_a, candidate_b):
        assert candidate.training == exp004.training
        assert candidate.data == exp004.data
        assert candidate.mixture == exp004.mixture
        assert candidate.training.full_schedule_steps == 9_156
        assert candidate.training.full_training_tokens == 300_023_808
        assert candidate.training.full_schedule_steps * candidate.training.effective_batch_tokens == candidate.training.full_training_tokens
        assert expected_full_sequences(candidate) == 585_984
        assert_physical_batch_control(candidate, 32, 2)
        with pytest.raises(RuntimeError, match="physical batch"):
            assert_physical_batch_control(candidate, 16, 4)


def test_exp007_full_path_dry_run_resume_cursor_and_scaling_milestones_remain_frozen() -> None:
    """Breaks if a near-cap preflight compresses the schedule or changes sequential stream cursor semantics."""
    for config_path in (Path("configs/exp007a.yaml"), Path("configs/exp007b.yaml")):
        config = load_config(config_path)
        assert full_run_milestones(config) == (0, 3_052, 6_104, 9_156)
        assert dry_run_plan(config, 0, 60) == (60, True)
        assert dry_run_plan(config, 60, 1) == (1, True)
        assert expected_run_state(config, 0, 60) == (60, 1_966_080, 3_840)
        assert expected_run_state(config, 60, 1) == (61, 1_998_848, 3_904)


def test_exp007_runner_accepts_only_the_exact_frozen_exp004_stream_and_dual_validations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Breaks if a near-cap candidate can substitute a rematerialized stream, tokenizer, or validation control."""
    config = load_config(Path("configs/exp007a.yaml"))
    artifact = tmp_path / "exp004"
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
        "experiment_id": "EXP-004", "preparation_mode": "full_stream",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "tokenizer": {"sha256": "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14", "vocab_size": 8192, "special_tokens": ["<|endoftext|>"]},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": 300_023_808, "train_token_count_including_final_target": 300_023_809, "train_examples": 585_984, "train_stream_file": stream.name, "train_stream_bytes": stream.stat().st_size, "train_stream_sha256": "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1", "non_cycled": True},
        "general_validation": {"file": "general_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7", "targets_sha256": "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"},
        "edu_validation": {"file": "edu_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a", "targets_sha256": "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4", "contamination_screened": True},
        "mixture": {**config.mixture, "actual_prediction_token_contributions": {"fineweb": 200_017_577, "fineweb_edu": 100_006_231}, "unique_document_count": 322_643},
    }
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("gibc_llm.full_run.sha256_file", lambda path: "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14" if Path(path).name == "tokenizer.json" else "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1")
    monkeypatch.setattr(
        "gibc_llm.full_run.tensor_sha256",
        lambda values: "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7" if int(values[0, 0]) == 0 else "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e" if int(values[0, 0]) == 1 else "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a" if int(values[0, 0]) == 2 else "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4",
    )
    assert load_full_run_artifact(artifact, config).manifest["experiment_id"] == "EXP-004"
    manifest["packed"]["train_stream_sha256"] = "not-the-frozen-stream"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen EXP-004 stream"):
        load_full_run_artifact(artifact, config)
