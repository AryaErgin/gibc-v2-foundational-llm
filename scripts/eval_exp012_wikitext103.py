"""Run frozen WikiText-103 rolling perplexity on selected EXP-012 checkpoint in CPU FP32."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gibc_llm.official_cpu_evaluation import AMENDMENT_COMMIT, CHECKPOINT_SHA256, EXPECTED_PARAMS, TOKENIZER_SHA256, enforce_cpu_isolation, validate_wikitext103_record


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


class _TextRequest:
    def __init__(self, text: str) -> None:
        self.args = (text,)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('Official CPU evaluation requires CUDA_VISIBLE_DEVICES="" before importing PyTorch.')
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Official WikiText-103 evaluation unexpectedly sees CUDA.")
    from datasets import load_dataset
    from lm_eval import utils as lm_eval_utils

    from gibc_llm.evaluation import CustomCausalLM
    from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
    from gibc_llm.tokenizer import load_tokenizer
    from gibc_llm.utils import load_config, sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp012.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an official result artifact: {args.output}")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256 or sha256_file(args.tokenizer) != TOKENIZER_SHA256:
        raise RuntimeError("Selected checkpoint or frozen tokenizer SHA-256 mismatch.")
    config = load_config(args.config)
    if config.model.context_length != 512:
        raise RuntimeError("WikiText-103 protocol requires max_seq_len 512.")
    model = DecoderOnlyTransformer(config.model).to(torch.device("cpu"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if parameter_breakdown(model).total != EXPECTED_PARAMS:
        raise RuntimeError("Selected checkpoint parameter-count mismatch.")
    enforce_cpu_isolation(torch, model)
    tokenizer = load_tokenizer(args.tokenizer)
    adapter = CustomCausalLM(model, tokenizer, torch.device("cpu"), batch_size=args.batch_size)
    started_at = _timestamp()
    started = time.perf_counter()
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", revision=WIKITEXT_REVISION)
    requests: list[_TextRequest] = []
    all_pairs: list[tuple[int, list[int], list[int]]] = []
    scored_tokens = 0
    utf8_bytes = 0
    for document_index, document in enumerate(dataset):
        text = document["text"]
        if not text:
            continue
        token_ids = tokenizer.encode(text).ids
        if not token_ids:
            continue
        request_index = len(requests)
        requests.append(_TextRequest(text))
        utf8_bytes += len(text.encode("utf-8"))
        document_scored = 0
        windows = lm_eval_utils.get_rolling_token_windows(
            token_list=token_ids,
            prefix_token=adapter.prefix_token_id,
            max_seq_len=512,
            context_len=1,
        )
        for window in windows:
            context, continuation = lm_eval_utils.make_disjoint_window(window)
            if not continuation:
                raise RuntimeError(f"WikiText-103 rolling window has empty continuation for document {document_index}.")
            all_pairs.append((request_index, context, continuation))
            document_scored += len(continuation)
        if document_scored != len(token_ids):
            raise RuntimeError(f"WikiText-103 token-accounting invariant failed for document {document_index}.")
        scored_tokens += document_scored
    if not requests or scored_tokens <= 0 or utf8_bytes <= 0:
        raise RuntimeError("WikiText-103 held-out slice produced no scoreable tokens.")
    scores = adapter._score_many(all_pairs, len(requests))
    total_nll = -sum(score for score, _ in scores)
    if not math.isfinite(total_nll):
        raise FloatingPointError("WikiText-103 summed NLL is non-finite.")
    mean_nll = total_nll / scored_tokens
    result = {
        "perplexity": math.exp(mean_nll),
        "bits_per_byte": total_nll / math.log(2.0) / utf8_bytes,
        "scored_tokens": scored_tokens,
        "documents": len(requests),
        "utf8_bytes": utf8_bytes,
        "summed_negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
    }
    ended_at = _timestamp()
    import datasets
    import lm_eval

    record = {
        "metadata": {
            "task": "wikitext103",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer": str(args.tokenizer),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "trainable_parameters": EXPECTED_PARAMS,
            "device": "cpu",
            "precision": "fp32",
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "lm_eval_version": lm_eval.__version__,
            "datasets_version": datasets.__version__,
            "torch_version": torch.__version__,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "context_length": 512,
            "amendment_commit": AMENDMENT_COMMIT,
            "dataset_repository": "wikitext",
            "dataset_revision": WIKITEXT_REVISION,
            "dataset_config": "wikitext-103-raw-v1",
            "dataset_split": "test",
            "rolling_context_len": 1,
            "no_added_bos_eos": True,
            "document_prefix": "one_eod_token",
            "command": sys.argv,
            "platform": platform.platform(),
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_seconds": time.perf_counter() - started,
            "no_pretrained_weights": True,
        },
        "result": result,
    }
    validate_wikitext103_record(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"task": "wikitext103", "status": "completed", "wall_seconds": record["metadata"]["wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
