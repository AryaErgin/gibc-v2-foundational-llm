"""Frozen CPU-FP32 gates and result validation for EXP-012 official evaluation."""

from __future__ import annotations

import math
import os
from typing import Any


CHECKPOINT_SHA256 = "cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c"
TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
EXPECTED_PARAMS = 49_860_480
AMENDMENT_COMMIT = "49bfe789fb8e6ebd23b00b5774f4f2e97ee1c464"
EXPECTED_LM_METRICS = {
    "hellaswag": ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"),
    "arc_easy": ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"),
    "piqa": ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"),
    "winogrande": ("acc,none", "acc_stderr,none"),
}


def enforce_cpu_isolation(torch_module: Any, model: Any) -> None:
    """Hard fail if an official CPU evaluator can observe or use CUDA."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('CPU official evaluation requires CUDA_VISIBLE_DEVICES="" before Python starts.')
    if torch_module.cuda.is_available():
        raise RuntimeError("CUDA isolation failure: torch.cuda.is_available() must be False.")
    parameters = list(model.parameters())
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise RuntimeError("CPU isolation failure: a model parameter is not on CPU.")
    if any(parameter.dtype != torch_module.float32 for parameter in parameters):
        raise RuntimeError("CPU official evaluation requires FP32 model parameters.")
    if not all(torch_module.isfinite(parameter).all() for parameter in parameters):
        raise RuntimeError("Official evaluation model contains a non-finite parameter.")


def _require_metadata(record: dict[str, Any], task: str) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Official evaluation artifact lacks metadata.")
    expected = {
        "task": task,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "trainable_parameters": EXPECTED_PARAMS,
        "device": "cpu",
        "precision": "fp32",
        "cuda_available": False,
        "lm_eval_version": "0.4.9.1",
        "num_fewshot": 0,
        "batch_size": 16,
        "context_length": 512,
        "amendment_commit": AMENDMENT_COMMIT,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            if name == "cuda_available":
                raise RuntimeError("CUDA isolation metadata failure.")
            raise RuntimeError(f"Official evaluation metadata mismatch for {name}: expected {value!r}, got {metadata.get(name)!r}.")
    return metadata


def _finite_metric(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Official evaluation has non-finite metric {name}.")
    return float(value)


def validate_lm_task_record(record: dict[str, Any], task: str) -> None:
    """Reject malformed, non-finite, or protocol-drifted lm-eval output."""
    if task not in EXPECTED_LM_METRICS:
        raise ValueError(f"Unsupported frozen lm-eval task {task!r}.")
    _require_metadata(record, task)
    try:
        metrics = record["raw_lm_eval_result"]["results"][task]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Official {task} artifact lacks raw lm-eval task metrics.") from exc
    for name in EXPECTED_LM_METRICS[task]:
        value = _finite_metric(metrics.get(name), name)
        if name.startswith("acc") and not name.startswith("acc_stderr") and not name.startswith("acc_norm_stderr") and not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Official evaluation accuracy {name} is outside [0, 1].")
        if "stderr" in name and value < 0.0:
            raise RuntimeError(f"Official evaluation stderr {name} is negative.")


def validate_wikitext103_record(record: dict[str, Any]) -> None:
    """Reject an incomplete or non-finite WikiText-103 rolling-PPL artifact."""
    _require_metadata(record, "wikitext103")
    result = record.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("WikiText-103 artifact lacks result data.")
    if _finite_metric(result.get("perplexity"), "perplexity") <= 0.0:
        raise RuntimeError("WikiText-103 perplexity must be positive.")
    if _finite_metric(result.get("bits_per_byte"), "bits_per_byte") <= 0.0:
        raise RuntimeError("WikiText-103 BPB must be positive.")
    token_count = result.get("scored_tokens")
    if not isinstance(token_count, int) or token_count <= 0:
        raise RuntimeError("WikiText-103 requires a positive scored-token count.")
