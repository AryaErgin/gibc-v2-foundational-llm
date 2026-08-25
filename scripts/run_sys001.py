"""Run one bounded SYS-001 systems phase; this entrypoint cannot request long-horizon training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gibc_llm.full_run import load_full_run_artifact
from gibc_llm.sys001 import (
    SYS001_SYSTEMS_CONTROL,
    assert_sys001_controls,
    phase_by_identifier,
    run_stability_validation,
    run_sys001_phase,
    sys001_phase_plan,
)
from gibc_llm.utils import atomic_json_write, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=[phase.identifier for phase in sys001_phase_plan()], required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp004-full"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/exp007b.yaml"))
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--timed-updates", type=int, default=100)
    parser.add_argument("--stability-reference", choices=[phase.identifier for phase in sys001_phase_plan()])
    args = parser.parse_args()
    if args.warmup_updates < 100 or args.timed_updates < 100:
        raise ValueError("SYS-001 requires at least 100 warmup and 100 timed updates.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SYS-001 requires a BF16-capable CUDA device.")
    if (args.run_dir / "summary.json").exists():
        raise RuntimeError("SYS-001 run directory already contains a summary; refusing to mix measurements.")
    config = load_config(args.config)
    assert_sys001_controls(config)
    artifact = load_full_run_artifact(args.artifact_dir, config)
    phase = phase_by_identifier(args.phase)
    result = {
        "experiment_id": "SYS-001",
        "systems_control": SYS001_SYSTEMS_CONTROL,
        "authorization_boundary": "bounded systems measurement only; no official benchmark or long-horizon training",
        "phase_result": run_sys001_phase(
            config,
            artifact,
            phase,
            torch.device("cuda"),
            warmup_updates=args.warmup_updates,
            timed_updates=args.timed_updates,
        ),
    }
    if args.stability_reference is not None:
        reference = phase_by_identifier(args.stability_reference)
        reference_result = run_stability_validation(config, artifact, reference, torch.device("cuda"))
        candidate_result = run_stability_validation(config, artifact, phase, torch.device("cuda"))
        result["stability_comparison"] = {
            "reference": reference_result,
            "candidate": candidate_result,
        }
        if reference_result["status"] == "COMPLETE" and candidate_result["status"] == "COMPLETE":
            result["stability_comparison"]["combined_loss_difference_candidate_minus_reference"] = (
                candidate_result["stability_validation"]["combined_loss"] - reference_result["stability_validation"]["combined_loss"]
            )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(args.run_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
