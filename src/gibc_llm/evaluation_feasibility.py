"""Scorer-free request accounting and CPU wall-time estimates for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


class LikelihoodInstance(Protocol):
    request_type: str
    args: tuple[str, str]
    task_name: str | None
    doc_id: int | None


@dataclass(frozen=True)
class RequestAccounting:
    examples: int
    continuations: int
    scored_tokens: int
    total_context_plus_continuation_tokens: int
    context_plus_continuation_lengths: list[int]
    event_context_lengths: list[int]


@dataclass(frozen=True)
class ScoringTimeEstimate:
    optimistic_seconds: float
    central_seconds: float
    pessimistic_seconds: float


def account_loglikelihood_requests(
    instances: Iterable[LikelihoodInstance],
    encode_pair: Callable[[str, str], tuple[list[int], list[int]]],
    *,
    max_context: int = 512,
) -> RequestAccounting:
    """Tokenize request pairs without invoking a model, scorer, or metric."""
    if max_context <= 0:
        raise ValueError("max_context must be positive.")
    examples: set[tuple[str | None, int | None] | tuple[str, int]] = set()
    continuations = 0
    scored_tokens = 0
    lengths: list[int] = []
    event_context_lengths: list[int] = []
    for index, instance in enumerate(instances):
        if instance.request_type not in {"loglikelihood", "multiple_choice"}:
            raise ValueError(f"CPU feasibility accounting accepts only likelihood requests, got {instance.request_type!r}.")
        context, continuation = instance.args
        context_ids, continuation_ids = encode_pair(context, continuation)
        if not continuation_ids:
            raise ValueError("Likelihood request has an empty tokenized continuation.")
        example_key: tuple[str | None, int | None] | tuple[str, int]
        if getattr(instance, "doc_id", None) is None:
            example_key = ("instance", index)
        else:
            example_key = (getattr(instance, "task_name", None), instance.doc_id)
        examples.add(example_key)
        continuations += 1
        scored_tokens += len(continuation_ids)
        total_length = len(context_ids) + len(continuation_ids)
        lengths.append(total_length)
        history_length = max(1, len(context_ids))
        for continuation_offset in range(len(continuation_ids)):
            event_context_lengths.append(min(max_context, history_length + continuation_offset))
    return RequestAccounting(
        examples=len(examples),
        continuations=continuations,
        scored_tokens=scored_tokens,
        total_context_plus_continuation_tokens=sum(lengths),
        context_plus_continuation_lengths=lengths,
        event_context_lengths=event_context_lengths,
    )


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * quantile)])


def estimate_scoring_seconds(
    *,
    event_context_lengths: Iterable[int],
    measured_scored_tokens_per_second: dict[int, float],
) -> ScoringTimeEstimate:
    """Estimate CPU work from measured synthetic per-token scoring rates.

    The central value selects the nearest measured context bucket per event.
    The optimistic value assumes the fastest measured bucket throughout. The
    pessimistic value applies the slowest measured bucket plus a 50% overhead
    allowance for real request-length mix, padding, and harness overhead.
    """
    if not measured_scored_tokens_per_second or any(rate <= 0.0 for rate in measured_scored_tokens_per_second.values()):
        raise ValueError("At least one positive synthetic scored-token rate is required.")
    contexts = list(event_context_lengths)
    if not contexts:
        return ScoringTimeEstimate(0.0, 0.0, 0.0)
    buckets = sorted(measured_scored_tokens_per_second)
    central = sum(1.0 / measured_scored_tokens_per_second[min(buckets, key=lambda bucket: abs(bucket - context))] for context in contexts)
    fastest = max(measured_scored_tokens_per_second.values())
    slowest = min(measured_scored_tokens_per_second.values())
    return ScoringTimeEstimate(
        optimistic_seconds=len(contexts) / fastest,
        central_seconds=central,
        pessimistic_seconds=1.5 * len(contexts) / slowest,
    )
