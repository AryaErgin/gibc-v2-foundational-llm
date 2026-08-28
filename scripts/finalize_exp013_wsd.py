"""Finalize only completed EXP-013 arm summaries; does not train or evaluate benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from gibc_llm.exp013 import assert_full_summary, classify, validate_stable_checkpoint
from gibc_llm.utils import atomic_json_write, load_config, sha256_file


def _final_edu_loss(summary: dict[str, Any]) -> float:
    records = summary.get("edu_validation_records", [])
    if not records or int(records[-1].get("step", -1)) != 9156:
        raise RuntimeError("Completed arm is missing its final frozen Edu validation result.")
    value = records[-1].get("loss")
    if not isinstance(value, float) or not math.isfinite(value):
        raise RuntimeError("Final frozen Edu validation loss is invalid.")
    return value


def _arm(summary_path: Path, config_path: Path, expected_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    if config.experiment_id != expected_id:
        raise RuntimeError("Config identity does not match the requested EXP-013 arm.")
    assert_full_summary(summary, expected_id)
    if summary.get("git_commit") is None:
        raise RuntimeError("Completed arm lacks the pre-training source/spec commit provenance.")
    return summary, config.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--wsd-summary", type=Path, required=True)
    parser.add_argument("--wsd-config", type=Path, default=Path("configs/exp013-wsd.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/exp013-wsd-ablation.json"))
    args = parser.parse_args()

    control, control_config = _arm(args.control_summary, Path("configs/exp013-cosine.yaml"), "EXP-013-C")
    wsd, wsd_config = _arm(args.wsd_summary, args.wsd_config, "EXP-013-W")
    result = classify(
        float(control["final_validation_loss"]),
        _final_edu_loss(control),
        float(wsd["final_validation_loss"]),
        _final_edu_loss(wsd),
    )
    stable_checkpoint = args.wsd_summary.parent / "checkpoints" / "checkpoint-step-8240.pt"
    result.update(
        {
            "experiment_id": "EXP-013",
            "primary_metric": "combined frozen validation loss at 300,023,808 prediction tokens",
            "control_summary_sha256": sha256_file(args.control_summary),
            "wsd_summary_sha256": sha256_file(args.wsd_summary),
            "control_config": control_config,
            "wsd_config": wsd_config,
            "control_runtime": {
                key: control[key]
                for key in ("wall_seconds", "mean_tokens_per_second", "final_tokens_per_second", "peak_allocated_bytes", "peak_reserved_bytes")
            },
            "wsd_runtime": {
                key: wsd[key]
                for key in ("wall_seconds", "mean_tokens_per_second", "final_tokens_per_second", "peak_allocated_bytes", "peak_reserved_bytes")
            },
            "data_provenance": {
                "tokenizer_sha256": control["tokenizer_sha256"],
                "data_manifest_sha256": control["data_manifest_sha256"],
                "source_stream_sha256": "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1",
            },
            "source_commits": {"control": control["git_commit"], "wsd": wsd["git_commit"]},
            "stable_stage_checkpoint": validate_stable_checkpoint(stable_checkpoint, load_config(args.wsd_config)),
        }
    )
    atomic_json_write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
