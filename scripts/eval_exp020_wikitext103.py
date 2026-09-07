"""Run future held-out WikiText-103 rolling PPL for the frozen EXP-020 terminal model."""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gibc_llm.exp020_official_evaluation import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PARAMS,
    EXPECTED_PREDICTION_TOKENS,
    EXPECTED_SELECTED_STEP,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_SUMMARY_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    WIKITEXT_REVISION,
    atomic_write_json,
    load_exp020_cpu_model,
    validate_runtime_provenance,
    validate_wikitext103_record,
)


class _TextRequest:
    def __init__(self, text: str) -> None:
        self.args = (text,)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an EXP-020 official result: {args.output}")
    validate_runtime_provenance(args.runtime_provenance)
    started_at = _timestamp()
    started = time.perf_counter()
    torch, config, _, _, adapter = load_exp020_cpu_model(args)
    # Dataset access is deliberately confined to this future execution script.
    from datasets import load_dataset
    from lm_eval import utils as lm_eval_utils

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", revision=WIKITEXT_REVISION)
    requests: list[_TextRequest] = []
    pairs: list[tuple[int, list[int], list[int]]] = []
    scored_tokens = 0
    utf8_bytes = 0
    for document_index, document in enumerate(dataset):
        text = document["text"]
        if not text:
            continue
        token_ids = adapter.tokenizer.encode(text).ids
        if not token_ids:
            continue
        request_index = len(requests)
        requests.append(_TextRequest(text))
        utf8_bytes += len(text.encode("utf-8"))
        document_scored = 0
        for window in lm_eval_utils.get_rolling_token_windows(token_list=token_ids, prefix_token=adapter.prefix_token_id, max_seq_len=512, context_len=1):
            context, continuation = lm_eval_utils.make_disjoint_window(window)
            if not continuation:
                raise RuntimeError(f"WikiText-103 empty continuation at document {document_index}.")
            pairs.append((request_index, context, continuation))
            document_scored += len(continuation)
        if document_scored != len(token_ids):
            raise RuntimeError(f"WikiText-103 every-token-once invariant failed at document {document_index}.")
        scored_tokens += document_scored
    if not requests or scored_tokens <= 0 or utf8_bytes <= 0:
        raise RuntimeError("WikiText-103 produced no scoreable held-out tokens.")
    scores = adapter._score_many(pairs, len(requests))
    total_nll = -sum(score for score, _ in scores)
    if not math.isfinite(total_nll):
        raise FloatingPointError("WikiText-103 summed NLL is non-finite.")
    mean_nll = total_nll / scored_tokens
    import datasets
    import lm_eval

    record = {
        "metadata": {
            "task": "wikitext103",
            "checkpoint": str(args.checkpoint),
            "tokenizer": str(args.tokenizer),
            "checkpoint_sha256": "95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89",
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "summary_sha256": EXPECTED_SUMMARY_SHA256,
            "trainable_parameters": EXPECTED_PARAMS,
            "selected_step": EXPECTED_SELECTED_STEP,
            "prediction_tokens": EXPECTED_PREDICTION_TOKENS,
            "training_source_commit": EXPECTED_SOURCE_COMMIT,
            "device": "cpu",
            "precision": "fp32",
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "datasets_version": datasets.__version__,
            "lm_eval_version": lm_eval.__version__,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "context_length": config.model.context_length,
            "dataset_repository": "wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "dataset_split": "test",
            "dataset_revision": WIKITEXT_REVISION,
            "rolling_context_len": 1,
            "document_prefix": "one_eod_token",
            "no_added_bos_eos": True,
            "command": sys.argv,
            "platform": platform.platform(),
            "started_at": started_at,
            "ended_at": _timestamp(),
            "wall_seconds": time.perf_counter() - started,
            "no_pretrained_weights": True,
        },
        "result": {
            "mean_negative_log_likelihood": mean_nll,
            "perplexity": math.exp(mean_nll),
            "bits_per_byte": total_nll / math.log(2.0) / utf8_bytes,
            "summed_negative_log_likelihood": total_nll,
            "scored_tokens": scored_tokens,
            "documents": len(requests),
            "utf8_bytes": utf8_bytes,
        },
    }
    validate_wikitext103_record(record)
    atomic_write_json(args.output, record)
    print("EXP-020 WikiText-103 official evaluator completed")


if __name__ == "__main__":
    main()
