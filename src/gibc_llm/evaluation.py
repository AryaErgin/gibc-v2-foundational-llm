"""Transparent lm-evaluation-harness likelihood adapter for the custom causal LM."""

from __future__ import annotations

import math
from typing import Any

import torch

from lm_eval.api.model import LM

from .generation import generate


class CustomCausalLM(LM):
    def __init__(self, model, tokenizer, device: torch.device, batch_size: int = 1) -> None:
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eod_token_id = tokenizer.token_to_id("<|endoftext|>")
        if self.eod_token_id is None:
            raise ValueError("Custom adapter requires the EXP-001 EOD token.")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    @torch.no_grad()
    def _score_tokens(self, prefix: list[int], continuation: list[int]) -> tuple[float, bool]:
        history = prefix or [self.eod_token_id]
        score = 0.0
        greedy = True
        for token in continuation:
            context = torch.tensor(history[-self.model.config.context_length :], dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model(context)[0, -1].float()
            score += float(torch.log_softmax(logits, dim=-1)[token])
            greedy = greedy and int(logits.argmax()) == token
            history.append(token)
        return score, greedy

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        outputs = []
        for request in requests:
            context, continuation = request.args
            outputs.append(self._score_tokens(self._encode(context), self._encode(continuation)))
        return outputs

    def loglikelihood_rolling(self, requests) -> list[float]:
        outputs = []
        for request in requests:
            text = request.args[0]
            score, _ = self._score_tokens([self.eod_token_id], self._encode(text))
            outputs.append(score)
        return outputs

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
