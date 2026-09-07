"""Benchmark-free identity and CPU-isolation preflight for future EXP-020 reporting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from gibc_llm.exp020_official_evaluation import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PARAMS,
    EXPECTED_PREDICTION_TOKENS,
    EXPECTED_SELECTED_STEP,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_SUMMARY_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    assert_fresh_official_output,
    assert_no_conflicting_evaluator,
    atomic_write_json,
    load_exp020_cpu_model,
    sha256_file,
    validate_runtime_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    assert_no_conflicting_evaluator()
    assert_fresh_official_output(args.output_dir)
    validate_runtime_provenance(args.runtime_provenance)
    torch, config, payload, model, _ = load_exp020_cpu_model(args)
    atomic_write_json(
        args.report,
        {
            "record_type": "EXP020_OFFICIAL_EVALUATION_BENCHMARK_FREE_PREFLIGHT",
            "status": "PASS",
            "benchmark_requests_executed": False,
            "dataset_materialized": False,
            "lm_eval_simple_evaluate_invoked": False,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "summary_sha256": EXPECTED_SUMMARY_SHA256,
            "trainable_parameters": EXPECTED_PARAMS,
            "selected_step": EXPECTED_SELECTED_STEP,
            "prediction_tokens": EXPECTED_PREDICTION_TOKENS,
            "training_source_commit": EXPECTED_SOURCE_COMMIT,
            "context_length": config.model.context_length,
            "device": "cpu",
            "precision": "fp32",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "checkpoint_state_keys": sorted(payload),
            "strict_state_dict_load": True,
            "adapter_direct_logit_equivalence": True,
            "model_parameter_finite": all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
            "runtime_provenance": str(args.runtime_provenance),
            "runtime_provenance_sha256": sha256_file(args.runtime_provenance),
        },
    )
    print("EXP-020 benchmark-free official-evaluation preflight: PASS")


if __name__ == "__main__":
    main()
