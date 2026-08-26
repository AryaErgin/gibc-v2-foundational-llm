"""EXP-012 controls: fresh 2.4B Recipe-v3 run and immutable prefix chain."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gibc_llm.full_run import assert_physical_batch_control, expected_full_sequences, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp012_config_freezes_recipe_v3_and_fresh_2_4b_schedule() -> None:
    config = load_config(Path("configs/exp012.yaml"))

    assert config.experiment_id == "EXP-012"
    assert (config.model.vocab_size, config.model.d_model, config.model.n_layers, config.model.n_heads, config.model.head_dim, config.model.d_ff) == (8192, 640, 9, 20, 32, 1728)
    assert config.model.activation == "swiglu"
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 49_860_480
    assert config.training.full_schedule_steps == 73_242
    assert config.training.full_training_tokens == 2_399_993_856
    assert config.training.full_training_tokens == config.training.full_schedule_steps * config.training.effective_batch_tokens
    assert expected_full_sequences(config) == 4_687_488
    assert full_run_milestones(config) == (0, 9_156, 18_312, 27_468, 36_624, 45_780, 54_936, 64_092, 73_242)
    assert config.mixture == {
        "target_prediction_tokens": {"fineweb": 1_599_995_904, "fineweb_edu": 799_997_952},
        "global_deduplication": "canonical_content_sha256",
        "sources": {
            "fineweb": {"repo": "HuggingFaceFW/fineweb", "config": "sample-10BT", "revision": "9bb295ddab0e05d785b879661af7260fed5140fc", "field": "text"},
            "fineweb_edu": {"repo": "HuggingFaceFW/fineweb-edu", "config": "default", "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "field": "text"},
        },
    }
    assert_physical_batch_control(config, 32, 2)


def test_exp012_prefix_verifier_hard_fails_on_one_byte_difference(tmp_path: Path) -> None:
    from gibc_llm.exp012 import verify_exp011_prefix

    stream = tmp_path / "stream.uint16"
    stream.write_bytes(b"abcdefghi")
    expected = hashlib.sha256(b"abcdef").hexdigest()

    assert verify_exp011_prefix(stream, byte_count=6, expected_sha256=expected) == expected
    with pytest.raises(RuntimeError, match="EXP-011 prefix SHA-256 mismatch"):
        verify_exp011_prefix(stream, byte_count=6, expected_sha256=hashlib.sha256(b"abcdeg").hexdigest())


def test_exp012_copies_the_existing_contamination_index_without_rebuilding(tmp_path: Path) -> None:
    from gibc_llm.exp012 import copy_exp011_benchmark_index

    source = tmp_path / "exp011" / "cache" / "benchmarks" / "benchmark-ngrams.sqlite"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"approved-benchmark-index")
    target = tmp_path / "exp012" / "cache" / "benchmarks" / "benchmark-ngrams.sqlite"

    copied_hash = copy_exp011_benchmark_index(source, target)
    assert target.read_bytes() == b"approved-benchmark-index"
    assert copied_hash == hashlib.sha256(b"approved-benchmark-index").hexdigest()
