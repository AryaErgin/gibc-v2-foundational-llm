"""Process-control tests for the guarded official-evaluation launcher."""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path

import pytest

from gibc_llm.evaluation_launch import EvaluationAlreadyRunning, run_guarded


def _sleeping_dummy(status_path: str, stdout_path: str, stderr_path: str, marker_path: str) -> None:
    """Use only a synthetic child process; it never imports or invokes lm-eval."""
    run_guarded(
        task="dummy",
        command=[
            sys.executable,
            "-c",
            "import pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text('started'); time.sleep(1.0); print('dummy complete')",
            marker_path,
        ],
        status_path=Path(status_path),
        stdout_path=Path(stdout_path),
        stderr_path=Path(stderr_path),
    )


def _wait_for_running(status_path: Path) -> dict[str, object]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status["state"] == "running":
                return status
        time.sleep(0.02)
    raise AssertionError("guard never wrote a running status artifact")


def _wait_for_marker(marker_path: Path) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if marker_path.exists():
            return
        time.sleep(0.02)
    raise AssertionError("synthetic evaluator child never started")


def test_guard_rejects_a_duplicate_live_task_and_writes_terminal_status_after_exit(tmp_path: Path) -> None:
    """A second launch cannot mistake a still-live evaluator for a completed run."""
    status_path = tmp_path / "dummy.status.json"
    stdout_path = tmp_path / "dummy.stdout.log"
    stderr_path = tmp_path / "dummy.stderr.log"
    marker_path = tmp_path / "dummy.started"
    process = multiprocessing.Process(
        target=_sleeping_dummy,
        args=(str(status_path), str(stdout_path), str(stderr_path), str(marker_path)),
    )
    process.start()
    try:
        running = _wait_for_running(status_path)
        assert running["terminal_at"] is None
        _wait_for_marker(marker_path)
        assert marker_path.read_text(encoding="utf-8") == "started"
        with pytest.raises(EvaluationAlreadyRunning, match="already has a live guarded evaluator"):
            run_guarded(
                task="dummy",
                command=[sys.executable, "-c", "print('must not run')"],
                status_path=status_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
    finally:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert process.exitcode == 0
    terminal = json.loads(status_path.read_text(encoding="utf-8"))
    assert terminal["state"] == "succeeded"
    assert terminal["returncode"] == 0
    assert terminal["terminal_at"] is not None
    assert "dummy complete" in stdout_path.read_text(encoding="utf-8")


def test_guard_replaces_only_a_stale_pid_and_records_it(tmp_path: Path) -> None:
    """A dead PID is not a permanent lock, but its replacement remains auditable."""
    status_path = tmp_path / "dummy.status.json"
    status_path.write_text(
        json.dumps({"task": "dummy", "state": "running", "supervisor_pid": 999_999_999, "terminal_at": None}),
        encoding="utf-8",
    )
    stdout_path = tmp_path / "dummy.stdout.log"
    stderr_path = tmp_path / "dummy.stderr.log"

    run_guarded(
        task="dummy",
        command=[sys.executable, "-c", "print('stale lock replaced')"],
        status_path=status_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    terminal = json.loads(status_path.read_text(encoding="utf-8"))
    assert terminal["state"] == "succeeded"
    assert terminal["replaced_stale_supervisor_pid"] == 999_999_999
    assert "stale lock replaced" in stdout_path.read_text(encoding="utf-8")
