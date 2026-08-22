"""Minimal greedy/top-k generation for checkpoint sanity checks."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.0, top_k: int | None = None, seed: int | None = None) -> str:
    encoded = tokenizer.encode(prompt).ids
    if not encoded:
        raise ValueError("Generation prompt must encode to at least one token.")
    device = next(model.parameters()).device
    token_ids = list(encoded)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = torch.tensor(token_ids[-model.config.context_length :], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(context)[0, -1].float()
        if temperature <= 0.0:
            next_token = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, min(top_k, logits.numel()))
                filtered = torch.full_like(logits, float("-inf"))
                filtered[indices] = values
                logits = filtered
            next_token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator))
        token_ids.append(next_token)
    if was_training:
        model.train()
    return tokenizer.decode(token_ids)
