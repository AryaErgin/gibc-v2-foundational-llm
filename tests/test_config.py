from pathlib import Path

from gibc_llm.utils import load_config


def test_exp001_config_preserves_approved_control_values() -> None:
    """Breaks if a fixed EXP-001 scientific configuration value changes."""
    config = load_config(Path("configs/exp001.yaml"))

    assert config.experiment_id == "EXP-001"
    assert config.model.vocab_size == 8192
    assert config.model.d_model == 256
    assert config.model.n_layers == 8
    assert config.model.n_heads == 8
    assert config.model.head_dim == 32
    assert config.model.d_ff == 1024
    assert config.model.context_length == 512
    assert config.training.effective_batch_tokens == 32768
    assert config.training.warmup_steps == 100
    assert config.training.full_training_tokens == 100_007_936
    assert config.data.dataset_revision == "9bb295ddab0e05d785b879661af7260fed5140fc"


def test_exp002_configuration_changes_only_horizon_and_token_budget() -> None:
    """Breaks if EXP-002 drifts a frozen EXP-001 scientific variable."""
    baseline = load_config(Path("configs/exp001.yaml"))
    scaled = load_config(Path("configs/exp002.yaml"))
    assert scaled.experiment_id == "EXP-002"
    assert scaled.model == baseline.model
    assert scaled.data == baseline.data
    assert scaled.training.full_schedule_steps == 9_156
    assert scaled.training.full_training_tokens == 300_023_808
    assert scaled.training.full_training_tokens == scaled.training.full_schedule_steps * scaled.training.effective_batch_tokens
    assert (scaled.training.default_microbatch_sequences, scaled.training.default_gradient_accumulation_steps) == (32, 2)
