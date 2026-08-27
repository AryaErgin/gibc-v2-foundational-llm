"""No-score request accounting used by the CPU evaluation feasibility probe."""

from __future__ import annotations

import pytest

from gibc_llm.evaluation_feasibility import account_loglikelihood_requests, estimate_scoring_seconds


class _Instance:
    def __init__(self, request_type: str, context: str, continuation: str) -> None:
        self.request_type = request_type
        self.args = (context, continuation)


def test_request_accounting_uses_only_joint_tokenization_not_model_scoring() -> None:
    """Breaks if feasibility accounting reaches a scorer or accesses benchmark labels."""
    calls: list[tuple[str, str]] = []

    def encode_pair(context: str, continuation: str) -> tuple[list[int], list[int]]:
        calls.append((context, continuation))
        return ([1] * len(context), [2] * len(continuation))

    accounting = account_loglikelihood_requests(
        [_Instance("loglikelihood", "abc", " de"), _Instance("multiple_choice", "q", " ans")],
        encode_pair,
    )

    assert calls == [("abc", " de"), ("q", " ans")]
    assert accounting.examples == 2
    assert accounting.continuations == 2
    assert accounting.scored_tokens == 7
    assert accounting.total_context_plus_continuation_tokens == 11
    assert accounting.context_plus_continuation_lengths == [6, 5]


def test_request_accounting_rejects_non_likelihood_instances_before_any_score() -> None:
    with pytest.raises(ValueError, match="only likelihood requests"):
        account_loglikelihood_requests([_Instance("generate_until", "prompt", "answer")], lambda _c, _t: ([1], [2]))


def test_cpu_time_estimate_uses_requested_pessimistic_central_optimistic_envelope() -> None:
    estimate = estimate_scoring_seconds(
        event_context_lengths=[120, 250, 511],
        measured_scored_tokens_per_second={128: 8.0, 256: 4.0, 512: 2.0},
    )

    assert estimate.optimistic_seconds == pytest.approx(3 / 8.0)
    assert estimate.central_seconds == pytest.approx(1 / 8.0 + 1 / 4.0 + 1 / 2.0)
    assert estimate.pessimistic_seconds == pytest.approx(1.5 * 3 / 2.0)
