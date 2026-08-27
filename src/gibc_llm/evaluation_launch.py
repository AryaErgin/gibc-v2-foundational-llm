"""Durable, single-supervisor process control for official evaluation commands."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


class EvaluationAlreadyRunning(RuntimeError):
    """Raised before a second supervisor can launch the same task."""


class EvaluationLaunchError(RuntimeError):
    """Raised for malformed or inconsistent evaluation status artifacts."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationLaunchError(f"Evaluation status artifact is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationLaunchError(f"Evaluation status artifact must contain a JSON object: {path}")
    return value


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Windows reports ERROR_INVALID_PARAMETER for a PID outside its valid
        # range instead of the POSIX-style ProcessLookupError.
        return False
    return True


def _lock_path(status_path: Path) -> Path:
    return status_path.with_suffix(status_path.suffix + ".lock")


def _lock_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _acquire_lock(status_path: Path, task: str) -> int | None:
    """Acquire a task-local exclusive lock, replacing only a dead owner."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    path = _lock_path(status_path)
    stale_pid: int | None = None
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            owner_pid = _lock_pid(path)
            if owner_pid is not None and _pid_is_alive(owner_pid):
                raise EvaluationAlreadyRunning(f"Task {task!r} already has a live guarded evaluator (supervisor PID {owner_pid}).")
            stale_pid = owner_pid
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return stale_pid
    raise EvaluationLaunchError(f"Could not acquire evaluation lock for task {task!r}.")


def _release_lock(status_path: Path) -> None:
    _lock_path(status_path).unlink(missing_ok=True)


def run_guarded(
    *,
    task: str,
    command: Sequence[str],
    status_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    environment: dict[str, str] | None = None,
) -> int:
    """Run one child evaluator and durably record its actual terminal state.

    The calling process is the sole supervisor. It owns a task-local lock and
    writes ``state=running`` before starting the child. Its terminal status is
    written only after ``wait()`` returns, so an outer transport must inspect
    this artifact rather than infer completion from a detached wrapper.
    """
    if not task:
        raise ValueError("Evaluation task must be non-empty.")
    if not command:
        raise ValueError("Guarded evaluation command must be non-empty.")

    stale_lock_pid = _acquire_lock(status_path, task)
    prior = _read_json(status_path)
    stale_status_pid: int | None = None
    if prior is not None:
        if prior.get("task") not in (None, task):
            _release_lock(status_path)
            raise EvaluationLaunchError(f"Status artifact {status_path} belongs to a different task.")
        if prior.get("state") == "running":
            supervisor_pid = int(prior.get("supervisor_pid", 0))
            if _pid_is_alive(supervisor_pid):
                _release_lock(status_path)
                raise EvaluationAlreadyRunning(
                    f"Task {task!r} already has a live guarded evaluator (supervisor PID {supervisor_pid})."
                )
            stale_status_pid = supervisor_pid

    replaced_stale_pid = stale_lock_pid if stale_lock_pid is not None else stale_status_pid
    status: dict[str, object] = {
        "task": task,
        "state": "running",
        "supervisor_pid": os.getpid(),
        "evaluator_pid": None,
        "command": list(command),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": _timestamp(),
        "terminal_at": None,
        "returncode": None,
        "replaced_stale_supervisor_pid": replaced_stale_pid,
    }
    _write_json_atomic(status_path, status)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    merged_environment = os.environ.copy()
    if environment is not None:
        merged_environment.update(environment)

    try:
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            stdout.write(f"[{_timestamp()}] guarded evaluation start task={task} command={list(command)!r}\n")
            stderr.write(f"[{_timestamp()}] guarded evaluation start task={task}\n")
            stdout.flush()
            stderr.flush()
            child = subprocess.Popen(list(command), stdout=stdout, stderr=stderr, env=merged_environment)
            status["evaluator_pid"] = child.pid
            _write_json_atomic(status_path, status)
            returncode = child.wait()
            status["returncode"] = returncode
            status["state"] = "succeeded" if returncode == 0 else "failed"
            status["terminal_at"] = _timestamp()
            stdout.write(f"[{_timestamp()}] guarded evaluation terminal state={status['state']} returncode={returncode}\n")
            stderr.write(f"[{_timestamp()}] guarded evaluation terminal state={status['state']} returncode={returncode}\n")
            return returncode
    except BaseException as exc:
        status["state"] = "launch_failed"
        status["terminal_at"] = _timestamp()
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json_atomic(status_path, status)
        _release_lock(status_path)
