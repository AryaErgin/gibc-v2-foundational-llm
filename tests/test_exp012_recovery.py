"""Synthetic process-control tests for stale EXP-012 CPU sequence recovery."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from gibc_llm.evaluation_launch import (
    EvaluationAlreadyRunning,
    assert_no_official_artifacts,
    recover_stale_sequence,
    run_guarded,
)
from gibc_llm.official_cpu_evaluation import artifact_path


def test_stale_running_sequence_with_dead_pids_is_archived_before_recovery(tmp_path: Path) -> None:
    status_dir = tmp_path / "status"
    logs_dir = tmp_path / "logs"
    sequence_path = status_dir / "sequence.status.json"
    hellaswag_path = status_dir / "hellaswag.status.json"
    suite_path = status_dir / "suite.status.json"
    status_dir.mkdir()
    logs_dir.mkdir()
    sequence_path.write_text(json.dumps({"state": "running", "supervisor_pid": 900_001, "evaluator_pid": 900_002}), encoding="utf-8")
    hellaswag_path.write_text(
        json.dumps(
            {
                "task": "hellaswag",
                "state": "running",
                "supervisor_pid": 900_003,
                "evaluator_pid": 900_004,
                "stdout_path": str(logs_dir / "hellaswag.stdout.log"),
                "stderr_path": str(logs_dir / "hellaswag.stderr.log"),
            }
        ),
        encoding="utf-8",
    )
    suite_path.write_text(json.dumps({"task": "suite", "state": "running", "supervisor_pid": 900_005}), encoding="utf-8")
    (logs_dir / "hellaswag.stdout.log").write_text("request construction only\n", encoding="utf-8")
    (logs_dir / "hellaswag.stderr.log").write_text("no result JSON\n", encoding="utf-8")

    archive = recover_stale_sequence(
        sequence_path,
        output_dir=tmp_path,
        pid_is_alive=lambda _pid: False,
        relevant_processes=lambda: [],
    )

    assert archive is not None
    assert not sequence_path.exists()
    assert not hellaswag_path.exists()
    assert suite_path.exists(), "the current top-level suite owner is not a sequence child"
    assert not (logs_dir / "hellaswag.stdout.log").exists()
    interruption = json.loads((archive / "interruption.json").read_text(encoding="utf-8"))
    assert interruption["classification"] == "INTERRUPTED_BY_TERMINAL_CLOSURE"
    assert interruption["valid_result_artifact_produced"] is False
    assert json.loads((archive / "status" / "hellaswag.status.json").read_text(encoding="utf-8"))["state"] == "running"
    assert (archive / "logs" / "hellaswag.stderr.log").read_text(encoding="utf-8") == "no result JSON\n"


def test_stale_running_sequence_refuses_recovery_when_a_recorded_pid_is_live(tmp_path: Path) -> None:
    status_path = tmp_path / "status" / "sequence.status.json"
    status_path.parent.mkdir()
    status_path.write_text(json.dumps({"state": "running", "supervisor_pid": 4321}), encoding="utf-8")

    with pytest.raises(EvaluationAlreadyRunning, match="live recorded PID 4321"):
        recover_stale_sequence(
            status_path,
            output_dir=tmp_path,
            pid_is_alive=lambda pid: pid == 4321,
            relevant_processes=lambda: [],
        )

    assert status_path.exists()
    assert not list((tmp_path / "history").glob("interrupted-*"))


def test_stale_running_sequence_refuses_recovery_when_a_relevant_evaluator_is_live(tmp_path: Path) -> None:
    status_path = tmp_path / "status" / "sequence.status.json"
    status_path.parent.mkdir()
    status_path.write_text(json.dumps({"state": "running", "supervisor_pid": 900_006}), encoding="utf-8")

    with pytest.raises(EvaluationAlreadyRunning, match="relevant evaluator process"):
        recover_stale_sequence(
            status_path,
            output_dir=tmp_path,
            pid_is_alive=lambda _pid: False,
            relevant_processes=lambda: [(4322, "python scripts/eval_exp012_cpu_task.py --task hellaswag")],
        )


def test_stale_sequence_recovery_permits_its_current_suite_supervisor(tmp_path: Path) -> None:
    status_dir = tmp_path / "status"
    sequence_path = status_dir / "sequence.status.json"
    suite_path = status_dir / "suite.status.json"
    status_dir.mkdir()
    sequence_path.write_text(json.dumps({"state": "running", "supervisor_pid": 900_007}), encoding="utf-8")
    suite_path.write_text(
        json.dumps({"task": "suite", "state": "running", "supervisor_pid": 4323, "evaluator_pid": os.getpid()}),
        encoding="utf-8",
    )

    archive = recover_stale_sequence(
        sequence_path,
        output_dir=tmp_path,
        pid_is_alive=lambda _pid: False,
        relevant_processes=lambda: [(4323, "python scripts/run_evaluation_guarded.py -- scripts/run_exp012_cpu_official_sequence.py")],
    )

    assert archive is not None
    assert suite_path.exists()


def test_sequence_preflight_refuses_existing_completed_official_artifact(tmp_path: Path) -> None:
    artifact = artifact_path(tmp_path, "hellaswag")
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite previous official artifact"):
        assert_no_official_artifacts(tmp_path, ("hellaswag", "arc_easy"))


def test_guard_refuses_a_live_evaluator_pid_even_when_the_supervisor_pid_is_dead(tmp_path: Path) -> None:
    status_path = tmp_path / "dummy.status.json"
    status_path.write_text(
        json.dumps({"task": "dummy", "state": "running", "supervisor_pid": 999_999_999, "evaluator_pid": os.getpid()}),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationAlreadyRunning, match="evaluator PID"):
        run_guarded(
            task="dummy",
            command=[sys.executable, "-c", "print('must not run')"],
            status_path=status_path,
            stdout_path=tmp_path / "dummy.stdout.log",
            stderr_path=tmp_path / "dummy.stderr.log",
        )
