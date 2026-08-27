"""Validation gates for CPU-only official EXP-012 result artifacts."""

from __future__ import annotations

import pytest
import torch

from gibc_llm.official_cpu_evaluation import enforce_cpu_isolation, validate_lm_task_record, validate_wikitext103_record


CHECKPOINT = "cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c"
TOKENIZER = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"


def _metadata(task: str) -> dict[str, object]:
    return {
        "task": task,
        "checkpoint_sha256": CHECKPOINT,
        "tokenizer_sha256": TOKENIZER,
        "trainable_parameters": 49_860_480,
        "device": "cpu",
        "precision": "fp32",
        "cuda_available": False,
        "lm_eval_version": "0.4.9.1",
        "num_fewshot": 0,
        "batch_size": 16,
        "context_length": 512,
        "amendment_commit": "49bfe789fb8e6ebd23b00b5774f4f2e97ee1c464",
    }


def test_lm_task_record_accepts_only_finite_frozen_cpu_result_metrics() -> None:
    record = {
        "metadata": _metadata("hellaswag"),
        "raw_lm_eval_result": {"results": {"hellaswag": {"acc,none": 0.25, "acc_stderr,none": 0.01, "acc_norm,none": 0.3, "acc_norm_stderr,none": 0.01}}},
    }

    validate_lm_task_record(record, "hellaswag")


def test_cpu_isolation_requires_hidden_cuda_and_fp32_cpu_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 2).float()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    enforce_cpu_isolation(torch, model)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        enforce_cpu_isolation(torch, model)


def test_lm_task_record_rejects_nonfinite_or_protocol_drift() -> None:
    record = {
        "metadata": _metadata("piqa"),
        "raw_lm_eval_result": {"results": {"piqa": {"acc,none": float("nan"), "acc_norm,none": 0.5}}},
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_lm_task_record(record, "piqa")

    record["raw_lm_eval_result"]["results"]["piqa"]["acc,none"] = 0.5
    record["metadata"]["cuda_available"] = True
    with pytest.raises(RuntimeError, match="CUDA isolation"):
        validate_lm_task_record(record, "piqa")


def test_wikitext_record_requires_positive_finite_metrics_and_token_count() -> None:
    record = {
        "metadata": _metadata("wikitext103"),
        "result": {"perplexity": 30.0, "bits_per_byte": 1.2, "scored_tokens": 100},
    }
    validate_wikitext103_record(record)

    record["result"]["scored_tokens"] = 0
    with pytest.raises(RuntimeError, match="positive scored-token"):
        validate_wikitext103_record(record)
