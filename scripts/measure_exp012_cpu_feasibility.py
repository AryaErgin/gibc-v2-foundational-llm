"""CPU-only, scorer-free feasibility measurement for frozen EXP-012 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

from gibc_llm.evaluation_feasibility import RequestAccounting, account_loglikelihood_requests, estimate_scoring_seconds, percentile


CHECKPOINT_SHA256 = "cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c"
TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
EXPECTED_PARAMS = 49_860_480
REQUEST_TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")


def _require_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('CPU feasibility requires CUDA_VISIBLE_DEVICES="" before Python starts.')


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _length_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": sum(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _synthetic_requests(context_length: int, batch_size: int, vocab_size: int) -> list[tuple[None, list[int], list[int]]]:
    if context_length > 512:
        raise ValueError("Synthetic context cannot exceed frozen context length 512.")
    return [
        (
            None,
            [((row * 97 + position) % (vocab_size - 1)) + 1 for position in range(context_length)],
            [((row * 193 + context_length) % (vocab_size - 1)) + 1],
        )
        for row in range(batch_size)
    ]


def _measure_synthetic(adapter: Any, *, context_length: int, batch_size: int, vocab_size: int, timed_batches: int) -> dict[str, float | int]:
    requests = _synthetic_requests(context_length, batch_size, vocab_size)
    adapter._loglikelihood_tokens(requests)  # warm-up; IDs are synthetic and no benchmark request is present.
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    for _ in range(timed_batches):
        adapter._loglikelihood_tokens(requests)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    scored_tokens = batch_size * timed_batches
    return {
        "context_tokens": context_length,
        "batch_size": batch_size,
        "timed_batches": timed_batches,
        "examples_per_second": scored_tokens / wall_seconds,
        "scored_tokens_per_second": scored_tokens / wall_seconds,
        "wall_seconds_per_batch": wall_seconds / timed_batches,
        "process_cpu_utilization_percent": 100.0 * cpu_seconds / wall_seconds,
    }


def _materialize_task(task_name: str, task_manager: Any, adapter: Any) -> RequestAccounting:
    """Build upstream lm-eval requests; deliberately do not create or call an evaluator/model scorer."""
    from lm_eval.tasks import get_task_dict

    task = get_task_dict([task_name], task_manager)[task_name]
    task.set_config(key="num_fewshot", value=0)
    task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False)
    return account_loglikelihood_requests(task.instances, adapter._encode_pair, max_context=adapter.max_length)


def _wiki103_accounting(tokenizer: Any) -> RequestAccounting:
    """Tokenize WikiText-103 test documents only; no model, likelihood, or PPL call occurs."""
    from datasets import load_dataset

    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="test",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    )
    total_tokens = 0
    event_contexts: list[int] = []
    documents = 0
    for document in dataset:
        text = document["text"]
        if not text:
            continue
        token_count = len(tokenizer.encode(text).ids)
        if token_count == 0:
            continue
        documents += 1
        total_tokens += token_count
        event_contexts.extend(min(512, position + 1) for position in range(token_count))
    return RequestAccounting(
        examples=documents,
        continuations=documents,
        scored_tokens=total_tokens,
        total_context_plus_continuation_tokens=total_tokens,
        context_plus_continuation_lengths=[total_tokens],
        event_context_lengths=event_contexts,
    )


def _accounting_record(accounting: RequestAccounting) -> dict[str, object]:
    return {
        "examples": accounting.examples,
        "continuations": accounting.continuations,
        "scored_tokens": accounting.scored_tokens,
        "context_plus_continuation_lengths": _length_summary(accounting.context_plus_continuation_lengths),
    }


def main() -> None:
    _require_cpu_only_environment()
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU feasibility process unexpectedly has CUDA available.")
    from lm_eval.tasks import TaskManager

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
    parser.add_argument("--timed-batches", type=int, default=3)
    args = parser.parse_args()
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Selected EXP-012 checkpoint SHA-256 mismatch.")
    if sha256_file(args.tokenizer) != TOKENIZER_SHA256:
        raise RuntimeError("Frozen tokenizer SHA-256 mismatch.")

    config = load_config(args.config)
    device = torch.device("cpu")
    model = DecoderOnlyTransformer(config.model).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    params = parameter_breakdown(model).total
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"Expected {EXPECTED_PARAMS} trainable parameters, got {params}.")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Selected checkpoint contains non-finite model tensor values.")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("CPU feasibility model contains a non-CPU parameter.")

    tokenizer = load_tokenizer(args.tokenizer)
    adapter = CustomCausalLM(model, tokenizer, device, batch_size=args.batch_size)
    synthetic = [
        _measure_synthetic(adapter, context_length=context_length, batch_size=args.batch_size, vocab_size=config.model.vocab_size, timed_batches=args.timed_batches)
        for context_length in (128, 256, 512)
    ]
    rates = {int(record["context_tokens"]): float(record["scored_tokens_per_second"]) for record in synthetic}

    task_manager = TaskManager()
    task_accounting = {task_name: _materialize_task(task_name, task_manager, adapter) for task_name in REQUEST_TASKS}
    wiki_accounting = _wiki103_accounting(tokenizer)
    estimates = {task_name: estimate_scoring_seconds(event_context_lengths=accounting.event_context_lengths, measured_scored_tokens_per_second=rates) for task_name, accounting in task_accounting.items()}
    estimates["wikitext-103"] = estimate_scoring_seconds(event_context_lengths=wiki_accounting.event_context_lengths, measured_scored_tokens_per_second=rates)

    result = {
        "measurement_kind": "synthetic_cpu_inference_and_no_score_request_accounting",
        "benchmark_scores_generated": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "trainable_parameters": params,
        "synthetic": synthetic,
        "thread_configuration": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "os_cpu_count": os.cpu_count(),
        },
        "peak_process_rss_mib": _rss_mib(),
        "task_request_accounting": {task_name: _accounting_record(accounting) for task_name, accounting in task_accounting.items()},
        "wikitext103_token_accounting": _accounting_record(wiki_accounting),
        "wall_time_estimates_seconds": {
            task_name: {
                "optimistic": estimate.optimistic_seconds,
                "central": estimate.central_seconds,
                "pessimistic": estimate.pessimistic_seconds,
            }
            for task_name, estimate in estimates.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
