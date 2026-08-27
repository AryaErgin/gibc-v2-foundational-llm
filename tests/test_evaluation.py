from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from gibc_llm.evaluation import CustomCausalLM


class _DeterministicModel:
    """A token transition model whose probabilities can be calculated by hand."""

    def __init__(self, vocab_size: int = 16, context_length: int = 4) -> None:
        self.config = SimpleNamespace(context_length=context_length)
        self.vocab_size = vocab_size
        self.seen_contexts: list[tuple[int, ...]] = []
        self.forward_calls = 0

    def to(self, device: torch.device) -> "_DeterministicModel":
        return self

    def eval(self) -> "_DeterministicModel":
        return self

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.seen_contexts.extend(tuple(row.tolist()) for row in input_ids)
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.vocab_size), dtype=torch.float32)
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 2.0)
        return logits


class _Request:
    def __init__(self, *args: str) -> None:
        self.args = args


def _byte_level_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.train_from_iterator(
        ["alpha beta gamma delta " * 500, "alphabet betamax " * 500],
        trainer=BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"], initial_alphabet=ByteLevel.alphabet(), show_progress=False),
    )
    return tokenizer


def _adapter(context_length: int = 4, batch_size: int = 1) -> tuple[CustomCausalLM, _DeterministicModel]:
    tokenizer = _byte_level_tokenizer()
    model = _DeterministicModel(vocab_size=tokenizer.get_vocab_size(), context_length=context_length)
    return CustomCausalLM(model, tokenizer, torch.device("cpu"), batch_size=batch_size), model


def test_loglikelihood_uses_joint_byte_level_bpe_suffix_not_independent_continuation() -> None:
    """Breaks if context/continuation are independently encoded at a BPE boundary."""
    adapter, _ = _adapter()
    context, continuation = "alph", "a beta gamma delta"

    context_ids, continuation_ids = adapter._encode_pair(context, continuation)
    independent = adapter.tok_encode(continuation)

    assert continuation_ids == adapter.tok_encode(context + continuation)[len(adapter.tok_encode(context)) :]
    assert continuation_ids != independent
    assert adapter.loglikelihood([_Request(context, continuation)])


def test_loglikelihood_moves_trailing_context_spaces_into_joint_continuation() -> None:
    """Breaks if trailing context spaces are scored/tokenized differently from TemplateLM."""
    adapter, _ = _adapter()
    context, continuation = "alpha ", "beta"

    context_ids, continuation_ids = adapter._encode_pair(context, continuation)

    assert context_ids == adapter.tok_encode("alpha")
    assert continuation_ids == adapter.tok_encode("alpha beta")[len(context_ids) :]


def test_empty_context_uses_eod_prefix_token() -> None:
    """Breaks if the first continuation token has no EOD/BOS predecessor."""
    adapter, model = _adapter()
    adapter.loglikelihood([_Request("", "alpha")])

    assert model.seen_contexts[0] == (adapter.eod_token_id,)


def test_loglikelihood_matches_manual_token_transition_logprobability() -> None:
    """Breaks if continuation token log-probabilities are shifted or summed incorrectly."""
    adapter, _ = _adapter()
    context_ids = [1]
    continuation_ids = [2, 3]
    log_probability_of_next = 2.0 - torch.log(torch.exp(torch.tensor(2.0)) + (adapter.model.vocab_size - 1)).item()

    score, greedy = adapter._score_tokens(context_ids, continuation_ids)

    assert score == pytest.approx(2 * log_probability_of_next)
    assert greedy


def test_rolling_loglikelihood_matches_bruteforce_and_scores_every_token_once() -> None:
    """Breaks if rolling windows skip, duplicate, or deny maximal preceding context to tokens."""
    adapter, model = _adapter(context_length=4)
    text = "alpha beta alpha beta alpha beta"
    token_ids = adapter.tok_encode(text)
    expected = 0.0
    history = [adapter.eod_token_id]
    log_probability_of_next = 2.0 - torch.log(torch.exp(torch.tensor(2.0)) + (adapter.model.vocab_size - 1)).item()
    for token in token_ids:
        expected += log_probability_of_next if token == (history[-1] + 1) % adapter.model.vocab_size else -torch.log(torch.exp(torch.tensor(2.0)) + (adapter.model.vocab_size - 1)).item()
        history.append(token)

    actual = adapter.loglikelihood_rolling([_Request(text)])[0]

    assert actual == pytest.approx(expected)
    assert len(model.seen_contexts) == len(token_ids)
    assert max(map(len, model.seen_contexts)) == adapter.model.config.context_length


def test_batched_loglikelihood_matches_serial_for_variable_and_long_continuations() -> None:
    """Breaks if batched causal scoring changes likelihoods, flags, or long-window semantics."""
    adapter, model = _adapter(context_length=4, batch_size=3)
    requests = [
        (None, [], [1]),
        (None, [1, 2, 3], [4, 5]),
        (None, [3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]),
        (None, [2], [3, 4, 5]),
    ]
    expected = [adapter._score_tokens(context, continuation) for _, context, continuation in requests]
    model.forward_calls = 0

    actual = adapter._loglikelihood_tokens(requests)

    assert actual == pytest.approx(expected)
    assert model.forward_calls < sum(len(continuation) for _, _, continuation in requests)


def test_batched_rolling_likelihood_matches_serial_beyond_a_model_window() -> None:
    """Breaks if batching rolling windows skips/duplicates a token beyond 512-token context."""
    text = "alpha beta " * 400
    serial, _ = _adapter(context_length=512, batch_size=1)
    batched, model = _adapter(context_length=512, batch_size=16)
    assert len(serial.tok_encode(text)) > 512

    expected = serial.loglikelihood_rolling([_Request(text)])[0]
    actual = batched.loglikelihood_rolling([_Request(text)])[0]

    assert actual == pytest.approx(expected, abs=1e-6)
    assert model.forward_calls < len(batched.tok_encode(text))


def test_evaluation_hard_fails_on_nonfinite_model_logits() -> None:
    """Official evaluation must stop rather than write a metric from NaN logits."""
    tokenizer = _byte_level_tokenizer()

    class _NonfiniteModel(_DeterministicModel):
        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), float("nan"))

    adapter = CustomCausalLM(_NonfiniteModel(vocab_size=tokenizer.get_vocab_size()), tokenizer, torch.device("cpu"))
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        adapter._loglikelihood_tokens([(None, [1], [2])])
