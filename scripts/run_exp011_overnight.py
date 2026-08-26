"""Safely sequence the authorized EXP-011 900M-to-1.5B continuation; exits at every failed gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def assert_900m_summary(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "DRY RUN / INCOMPLETE TRAINING",
        "final_step": 27_468,
        "prediction_tokens": 900_071_424,
        "next_sequence_index": 1_757_952,
        "parameter_count": 49_860_480,
        "microbatch_sequences": 32,
        "gradient_accumulation_steps": 2,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("EXP-011 900M phase summary does not satisfy the authorized resume boundary.")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    run(["powershell", "-NoProfile", "-Command", f"Wait-Process -Id {args.wait_pid}"])
    assert_900m_summary(Path("artifacts/exp011-full/summary.json"))
    run([
        sys.executable,
        "scripts/prepare_exp011.py",
        "--config",
        "configs/exp011.yaml",
        "--artifact-dir",
        "artifacts/exp011-full-data",
        "--exp006-artifact-dir",
        "artifacts/exp006-full",
    ])
    run([
        sys.executable,
        "-c",
        "from pathlib import Path; from gibc_llm.full_run import load_full_run_artifact; from gibc_llm.utils import load_config; artifact=load_full_run_artifact(Path('artifacts/exp011-full-data'), load_config('configs/exp011.yaml')); assert artifact.manifest['experiment_id']=='EXP-011' and len(artifact.train)==2929728",
    ])
    run([
        sys.executable,
        "scripts/train_exp001_full.py",
        "--config",
        "configs/exp011.yaml",
        "--artifact-dir",
        "artifacts/exp011-full-data",
        "--run-dir",
        "artifacts/exp011-full-continue",
        "--resume",
        "artifacts/exp011-full/checkpoints/checkpoint-step-27468.pt",
    ])


if __name__ == "__main__":
    main()
