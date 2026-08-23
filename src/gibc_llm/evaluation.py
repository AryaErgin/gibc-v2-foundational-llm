"""lm-evaluation-harness v0.4.9.1 adapter for the custom causal LM."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from lm_eval.api.model import TemplateLM
from lm_eval import utils as lm_eval_utils

from .generation import generate


class CustomCausalLM(TemplateLM):
    """Causal likelihood adapter without a pretrained-model wrapper.

    ``TemplateLM`` owns request tokenization. Its ``_encode_pair`` moves trailing
    context whitespace into the continuation and takes continuation IDs as the
    suffix of the joint byte-level-BPE encoding, matching lm-eval 0.4.9.1.
    """

    def __init__(self, model: Any, tokenizer: Any, device: torch.device, batch_size: int = 1) -> None:
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eod_token_id = tokenizer.token_to_id("<|endoftext|>")
        if self.eod_token_id is None:
            raise ValueError("Custom adapter requires the EXP-001 EOD token.")

    @property
    def eot_token_id(self) -> int:
        return self.eod_token_id

    @property
    def max_length(self) -> int:
        return self.model.config.context_length

    def tok_encode(self, text: str, **kwargs: Any) -> list[int]:
        # EXP-001 has no automatic special tokens. kwargs are accepted solely for
        # TemplateLM interface compatibility and never alter the tokenizer.
        return self.tokenizer.encode(text).ids

    @torch.no_grad()
    def _score_tokens(self, prefix: list[int], continuation: list[int]) -> tuple[float, bool]:
        history = list(prefix) or [self.prefix_token_id]
        score = 0.0
        greedy = True
        for token in continuation:
            context = torch.tensor(history[-self.max_length :], dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model(context)[0, -1].float()
            score += float(torch.log_softmax(logits, dim=-1)[token])
            greedy = greedy and int(logits.argmax()) == token
            history.append(token)
        return score, greedy

    @torch.no_grad()
    def _score_many(self, pairs: Iterable[tuple[int, list[int], list[int]]], output_count: int) -> list[tuple[float, bool]]:
        """Batch exact causal token events; right padding follows each scored position."""
        scores = [0.0] * output_count
        greedy = [True] * output_count
        events: list[tuple[int, list[int], int]] = []
        batch_size = max(1, int(self.batch_size))

        def score_batch(batch: list[tuple[int, list[int], int]]) -> None:
            if not batch:
                return
            lengths = torch.tensor([len(context) for _, context, _ in batch], device=self.device)
            width = int(lengths.max())
            input_ids = torch.full((len(batch), width), self.prefix_token_id, dtype=torch.long, device=self.device)
            for row, (_, context, _) in enumerate(batch):
                input_ids[row, : len(context)] = torch.tensor(context, dtype=torch.long, device=self.device)
            logits = self.model(input_ids).float()
            row_ids = torch.arange(len(batch), device=self.device)
            final_logits = logits[row_ids, lengths - 1]
            target_ids = torch.tensor([token for _, _, token in batch], device=self.device)
            token_scores = torch.log_softmax(final_logits, dim=-1)[row_ids, target_ids]
            predictions = final_logits.argmax(dim=-1)
            for event, value, prediction in zip(batch, token_scores, predictions, strict=True):
                output_index, _, token = event
                scores[output_index] += float(value)
                greedy[output_index] = greedy[output_index] and int(prediction) == token

        for output_index, prefix, continuation in pairs:
            history = list(prefix) or [self.prefix_token_id]
            for token in continuation:
                events.append((output_index, history[-self.max_length :], token))
                history.append(token)
                if len(events) == batch_size:
                    score_batch(events)
                    events.clear()
        score_batch(events)
        return list(zip(scores, greedy, strict=True))

    def _loglikelihood_tokens(self, requests, **kwargs: Any) -> list[tuple[float, bool]]:
        """Score TemplateLM triplets after its canonical joint tokenization."""
        pairs = []
        for index, (_, context_ids, continuation_ids) in enumerate(requests):
            if not continuation_ids:
                raise ValueError("lm-eval causal likelihood requests require a non-empty continuation.")
            pairs.append((index, context_ids, continuation_ids))
        return self._score_many(pairs, len(pairs))

    def loglikelihood_rolling(self, requests) -> list[float]:
        def pairs() -> Iterable[tuple[int, list[int], list[int]]]:
            for index, request in enumerate(requests):
                text = request.args[0]
                windows = map(lm_eval_utils.make_disjoint_window, lm_eval_utils.get_rolling_token_windows(token_list=self.tok_encode(text), prefix_token=self.prefix_token_id, max_seq_len=self.max_length, context_len=1))
                yield from ((index, context, continuation) for context, continuation in windows)
        return [score for score, _ in self._score_many(pairs(), len(requests))]

    def generate_until(self, requests) -> list[str]:
        outputs = []
        for request in requests:
            context, kwargs = request.args
            text = generate(
                self.model,
                self.tokenizer,
                context,
                max_new_tokens=int(kwargs.get("max_gen_toks", 32)),
                temperature=0.0,
            )
            continuation = text[len(context) :]
            for stop in kwargs.get("until", []):
                continuation = continuation.split(stop, 1)[0]
            outputs.append(continuation)
        return outputs

    @property
    def tokenizer_name(self) -> str:
        return "exp001-byte-level-bpe-8192"
