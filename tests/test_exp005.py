import json
from pathlib import Path

import pytest
import torch

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


FROZEN_TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
FROZEN_STREAM_SHA256 = "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1"
GENERAL_INPUTS_SHA256 = "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7"
GENERAL_TARGETS_SHA256 = "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"
EDU_INPUTS_SHA256 = "cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a"
EDU_TARGETS_SHA256 = "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4"


def test_exp005_candidates_have_approved_real_model_counts_and_fixed_controls() -> None:
    """Breaks if either allocation changes a frozen training control or its real parameter count."""
    exp004 = load_config(Path("configs/exp004.yaml"))
    deep = load_config(Path("configs/exp005a.yaml"))
    wide = load_config(Path("configs/exp005b.yaml"))

    assert deep.experiment_id == "EXP-005A"
    assert wide.experiment_id == "EXP-005B"
    assert (deep.model.d_model, deep.model.n_layers, deep.model.n_heads, deep.model.head_dim, deep.model.d_ff) == (256, 24, 8, 32, 1024)
    assert (wide.model.d_model, wide.model.n_layers, wide.model.n_heads, wide.model.head_dim, wide.model.d_ff) == (384, 10, 12, 32, 1536)
    assert parameter_breakdown(DecoderOnlyTransformer(deep.model)).total == 20_984_064
    assert parameter_breakdown(DecoderOnlyTransformer(wide.model)).total == 20_848_512
    assert parameter_breakdown(DecoderOnlyTransformer(deep.model)).total - parameter_breakdown(DecoderOnlyTransformer(wide.model)).total == 135_552
    assert 0.0064 < 135_552 / 20_848_512 < 0.0066
    assert deep.training == exp004.training == wide.training
    assert deep.data == exp004.data == wide.data
    assert deep.mixture == exp004.mixture == wide.mixture
    for candidate in (deep, wide):
        assert candidate.training.full_schedule_steps == 9_156
        assert candidate.training.full_training_tokens == 300_023_808
        assert candidate.training.full_schedule_steps * candidate.training.effective_batch_tokens == candidate.training.full_training_tokens
        assert expected_full_sequences(candidate) == 585_984
        assert_physical_batch_control(candidate, 32, 2)
        with pytest.raises(RuntimeError, match="physical batch"):
            assert_physical_batch_control(candidate, 16, 4)


def _exp004_fixture(tmp_path: Path, config: object) -> Path:
    artifact = tmp_path / "exp004"
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
        "experiment_id": "EXP-004",
        "preparation_mode": "full_stream",
        "dataset": {"repo": config.data.dataset_repo, "config": config.data.dataset_config, "revision": config.data.dataset_revision, "field": config.data.text_field},
        "tokenizer": {"sha256": FROZEN_TOKENIZER_SHA256, "vocab_size": 8192, "special_tokens": ["<|endoftext|>"]},
        "packed": {"representation": "one-dimensional uint16 token stream with on-demand torch.long 513-token views", "storage_dtype": "uint16", "context_length": 512, "prediction_tokens_per_example": 512, "train_prediction_tokens": 300_023_808, "train_token_count_including_final_target": stored, "train_examples": 585_984, "train_stream_file": stream.name, "train_stream_bytes": stream.stat().st_size, "train_stream_sha256": FROZEN_STREAM_SHA256, "non_cycled": True},
        "general_validation": {"file": "general_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": GENERAL_INPUTS_SHA256, "targets_sha256": GENERAL_TARGETS_SHA256},
        "edu_validation": {"file": "edu_validation.pt", "prediction_tokens": 131_072, "inputs_sha256": EDU_INPUTS_SHA256, "targets_sha256": EDU_TARGETS_SHA256, "contamination_screened": True},
        "mixture": {**config.mixture, "actual_prediction_token_contributions": {"fineweb": 200_017_577, "fineweb_edu": 100_006_231}, "unique_document_count": 322_643},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


def test_exp005_reuses_only_the_exact_frozen_exp004_stream_and_validations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Breaks if a candidate can consume a rematerialized/mismatched stream or changed dual validations."""
    config = load_config(Path("configs/exp005a.yaml"))
    artifact = _exp004_fixture(tmp_path, config)

    monkeypatch.setattr(
        "gibc_llm.full_run.sha256_file",
        lambda path: FROZEN_TOKENIZER_SHA256 if Path(path).name == "tokenizer.json" else FROZEN_STREAM_SHA256,
    )
    monkeypatch.setattr(
        "gibc_llm.full_run.tensor_sha256",
        lambda values: GENERAL_INPUTS_SHA256 if int(values[0, 0]) == 0 else GENERAL_TARGETS_SHA256 if int(values[0, 0]) == 1 else EDU_INPUTS_SHA256 if int(values[0, 0]) == 2 else EDU_TARGETS_SHA256,
    )
    loaded = load_full_run_artifact(artifact, config)
    assert len(loaded.train) == 585_984
    assert loaded.manifest["packed"]["train_stream_sha256"] == FROZEN_STREAM_SHA256

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packed"]["train_stream_sha256"] = "not-the-exp004-stream"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen EXP-004 stream"):
        load_full_run_artifact(artifact, config)


def test_exp005_dry_run_and_resume_cursor_arithmetic_stays_on_9156_horizon() -> None:
    """Breaks if a bounded preflight compresses the schedule or loses the exact sequential cursor."""
    config = load_config(Path("configs/exp005b.yaml"))
    assert dry_run_plan(config, 0, 60) == (60, True)
    assert dry_run_plan(config, 60, 1) == (1, True)
    assert dry_run_plan(config, 0, None) == (9_156, False)
    assert expected_run_state(config, 0, 60) == (60, 1_966_080, 3_840)
    assert expected_run_state(config, 60, 1) == (61, 1_998_848, 3_904)


def test_exp005_declares_the_same_internal_validation_and_checkpoint_milestones() -> None:
    """Breaks if either full candidate omits an equal-token scaling-curve milestone."""
    for config_path in (Path("configs/exp005a.yaml"), Path("configs/exp005b.yaml")):
        assert full_run_milestones(load_config(config_path)) == (0, 3_052, 6_104, 9_156)
