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
    assert config.training.full_training_tokens == 100_000_000
