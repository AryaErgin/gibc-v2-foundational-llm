"""EXP-015 fixed-example permutation controls; no benchmark code is invoked."""

from pathlib import Path

import numpy as np
import pytest

from gibc_llm.data import Document, stable_document_id, write_token_stream


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _Tokenizer:
    def encode(self, text: str) -> _Encoded:
        return _Encoded({"broad": [11, 12], "broad_2": [13, 14], "edu": [21, 22]}[text])


def _document(text: str) -> Document:
    return Document(document_id=stable_document_id(text), text=text, split="train")


def test_exp015_replay_sidecar_preserves_stream_and_original_prediction_attribution(tmp_path: Path) -> None:
    """Breaks if sidecar replay changes token bytes or labels prediction positions incorrectly."""
    from gibc_llm.exp004 import GlobalDeduplicatedTokenMixer
    from gibc_llm.exp015 import replay_source_attribution

    mixer = GlobalDeduplicatedTokenMixer(
        {"fineweb": iter([_document("broad"), _document("broad_2")]), "fineweb_edu": iter([_document("edu")])},
        _Tokenizer(),
        eod_id=99,
        target_prediction_tokens={"fineweb": 3, "fineweb_edu": 3},
        stored_token_count=7,
    )
    stream = write_token_stream(tmp_path / "stream.uint16", list(mixer), token_count=7, context_length=2)
    replay = GlobalDeduplicatedTokenMixer(
        {"fineweb": iter([_document("broad"), _document("broad_2")]), "fineweb_edu": iter([_document("edu")])},
        _Tokenizer(),
        eod_id=99,
        target_prediction_tokens={"fineweb": 3, "fineweb_edu": 3},
        stored_token_count=7,
    )

    record = replay_source_attribution(stream.path, 7, replay.iter_with_sources(), tmp_path / "sources.uint8")

    assert record["stream_match"] is True
    assert record["prediction_token_contributions"] == {"fineweb": 3, "fineweb_edu": 3}
    labels = np.memmap(tmp_path / "sources.uint8", mode="r", dtype=np.uint8, shape=(7,))
    # The first stored token is not a prediction target. The remaining labels
    # follow the same producer-source semantics as the original mixer.
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 0]


def test_exp015_fixed_window_schedule_is_a_permutation_with_only_tail_block_swap() -> None:
    """Breaks if the treatment changes membership, phase one, or block-relative order."""
    from gibc_llm.exp015 import FixedExampleSchedule

    # 12 windows of four prediction tokens: phase one is eight, tails are two
    # windows each. Target labels make tail windows 8/9 low and 10/11 high.
    labels = np.array([0] + ([0] * 40) + ([1] * 8), dtype=np.uint8)
    schedules = FixedExampleSchedule.build(labels, context_length=4, phase1_windows=8, block_windows=2)

    assert schedules.edu_counts.tolist() == [0] * 10 + [4, 4]
    assert schedules.low.tolist() == [8, 9]
    assert schedules.high.tolist() == [10, 11]
    assert schedules.static.tolist() == list(range(12))
    assert schedules.cooldown_edu.tolist() == list(range(8)) + [8, 9, 10, 11]
    assert schedules.precooldown_edu.tolist() == list(range(8)) + [10, 11, 8, 9]
    assert schedules.treatment_contrast == pytest.approx(1.0)
    schedules.assert_integrity()


def test_exp015_schedule_rejects_insufficient_tail_separation() -> None:
    """Breaks if a weak Edu treatment can be silently scheduled."""
    from gibc_llm.exp015 import FixedExampleSchedule

    labels = np.zeros(1 + 12 * 4, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="treatment separation"):
        FixedExampleSchedule.build(labels, context_length=4, phase1_windows=8, block_windows=2)


def test_exp015_adamw_retention_matches_fixed_wsd_phase_semantics() -> None:
    """Breaks if retention uses a different WSD step convention or cooldown boundary."""
    from gibc_llm.exp015 import adamw_retention_diagnostic

    report = adamw_retention_diagnostic()
    assert report["max_step"] == 8251
    assert report["boundary_values"]["8241"] > report["boundary_values"]["8240"]
    assert report["phase_means"]["phase2"] == pytest.approx(0.973, abs=0.003)
    assert report["phase_means"]["phase3"] == pytest.approx(0.558, abs=0.003)


def test_exp015_schedule_cursor_state_preserves_arm_hash_and_cursor() -> None:
    """Breaks if a future checkpoint can lose the selected fixed-example schedule."""
    from gibc_llm.exp015 import FixedExampleSchedule, schedule_cursor_state

    labels = np.array([0] + ([0] * 40) + ([1] * 8), dtype=np.uint8)
    schedule = FixedExampleSchedule.build(labels, context_length=4, phase1_windows=8, block_windows=2)
    state = schedule_cursor_state(schedule, cursor=8, arm="C")

    assert state["mechanism"] == "fixed_example_index_permutation"
    assert state["arm"] == "C"
    assert state["next_schedule_cursor"] == 8
    assert state["schedule_sha256"] == schedule.schedule_hashes()["precooldown_edu"]
