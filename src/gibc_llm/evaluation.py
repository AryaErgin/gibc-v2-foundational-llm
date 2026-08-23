"""lm-evaluation-harness v0.4.9.1 adapter for the custom causal LM."""

from __future__ import annotations

from typing import Any

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

    def _loglikelihood_tokens(self, requests, **kwargs: Any) -> list[tuple[float, bool]]:
        """Score TemplateLM triplets after its canonical joint tokenization."""
        outputs: list[tuple[float, bool]] = []
        for _, context_ids, continuation_ids in requests:
            if not continuation_ids:
                raise ValueError("lm-eval causal likelihood requests require a non-empty continuation.")
            if len(continuation_ids) > self.max_length:
                raise ValueError("lm-eval continuation exceeds the EXP-001 512-token context length.")
            outputs.append(self._score_tokens(context_ids, continuation_ids))
        return outputs

    def loglikelihood_rolling(self, requests) -> list[float]:
        outputs = []
        for request in requests:
            text = request.args[0]
            windows = map(
                lm_eval_utils.make_disjoint_window,
                lm_eval_utils.get_rolling_token_windows(
                    token_list=self.tok_encode(text),
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            )
            outputs.append(sum(self._score_tokens(context, continuation)[0] for context, continuation in windows))
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
