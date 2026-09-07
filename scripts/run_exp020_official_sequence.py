"""Future-only sequential EXP-020 official reporting runner; never use for preflight."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from gibc_llm.evaluation_launch import run_guarded
from gibc_llm.exp020_official_evaluation import (
    EXP020_OFFICIAL_TASKS,
    artifact_path,
    assert_fresh_official_output,
    assert_no_conflicting_evaluator,
    atomic_write_json,
    sha256_file,
    validate_lm_task_record,
    validate_runtime_provenance,
    validate_wikitext103_record,
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('EXP-020 official sequence requires CUDA_VISIBLE_DEVICES="".')
    assert_no_conflicting_evaluator(allowed_pids={os.getpid(), os.getppid()})
    assert_fresh_official_output(args.output_dir)
    validate_runtime_provenance(args.runtime_provenance)
    status = args.output_dir / "status" / "sequence.status.json"
    sequence: dict[str, object] = {
        "state": "running",
        "started_at": _timestamp(),
        "terminal_at": None,
        "tasks": [],
        "selection_before_benchmark_exposure": True,
    }
    atomic_write_json(status, sequence)
    try:
        for task in EXP020_OFFICIAL_TASKS:
            artifact = artifact_path(args.output_dir, task)
            script = "scripts/eval_exp020_wikitext103.py" if task == "wikitext103" else "scripts/eval_exp020_cpu_task.py"
            command = [
                args.python, script, "--config", str(args.config), "--checkpoint", str(args.checkpoint),
                "--tokenizer", str(args.tokenizer), "--summary", str(args.summary),
                "--runtime-provenance", str(args.runtime_provenance), "--output", str(artifact), "--batch-size", "16",
            ]
            if task != "wikitext103":
                command.extend(["--task", task])
            returncode = run_guarded(
                task=f"exp020_{task}", command=command,
                status_path=args.output_dir / "status" / f"{task}.status.json",
                stdout_path=args.output_dir / "logs" / f"{task}.stdout.log",
                stderr_path=args.output_dir / "logs" / f"{task}.stderr.log",
                environment={"CUDA_VISIBLE_DEVICES": ""},
            )
            if returncode != 0 or not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"EXP-020 official task {task} failed or did not produce a result artifact.")
            import json
            record = json.loads(artifact.read_text(encoding="utf-8"))
            if task == "wikitext103":
                validate_wikitext103_record(record)
            else:
                validate_lm_task_record(record, task)
            sequence["tasks"].append({"task": task, "status": "succeeded", "artifact": str(artifact), "sha256": sha256_file(artifact)})  # type: ignore[index]
            atomic_write_json(status, sequence)
    except BaseException as exc:
        sequence.update({"state": "failed", "terminal_at": _timestamp(), "error": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(status, sequence)
        raise
    sequence.update({"state": "succeeded", "terminal_at": _timestamp()})
    atomic_write_json(status, sequence)


if __name__ == "__main__":
    main()
