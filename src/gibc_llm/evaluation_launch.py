"""Durable, single-supervisor process control for official evaluation commands."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence


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


def _recorded_pid(record: dict[str, object], field: str, status_path: Path) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationLaunchError(f"Running status {status_path} has an invalid {field}: {value!r}.")
    return value


def _timestamp_filename(value: object) -> str:
    return str(value).replace(":", "-").replace("+", "_")


def _new_history_directory(history: Path, prefix: str) -> Path:
    history.mkdir(parents=True, exist_ok=True)
    base = history / f"{prefix}-{_timestamp_filename(_timestamp())}"
    target = base
    suffix = 1
    while target.exists():
        target = history / f"{base.name}.{suffix}"
        suffix += 1
    target.mkdir()
    return target


def _move_status_logs_to_archive(record: dict[str, object], archive: Path) -> list[str]:
    archived: list[str] = []
    logs = archive / "logs"
    for field in ("stdout_path", "stderr_path"):
        raw_path = record.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        source = Path(raw_path)
        if not source.is_file():
            continue
        logs.mkdir(parents=True, exist_ok=True)
        target = logs / source.name
        suffix = 1
        while target.exists():
            target = logs / f"{source.stem}.{suffix}{source.suffix}"
            suffix += 1
        os.replace(source, target)
        archived.append(str(target))
    return archived


def _archive_interrupted_guarded_attempt(status_path: Path, prior: dict[str, object]) -> Path:
    """Move one stale guarded attempt, including its logs, into durable history."""
    archive = _new_history_directory(status_path.parent / "history", "interrupted")
    status_target = archive / "status"
    status_target.mkdir()
    os.replace(status_path, status_target / status_path.name)
    _move_status_logs_to_archive(prior, archive)
    return archive


def _relevant_evaluator_processes() -> list[tuple[int, str]]:
    """Return actual EXP-012 evaluator processes; failure to inspect is unsafe."""
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise EvaluationLaunchError("Could not inspect the process table before stale-sequence recovery.") from exc
    markers = (
        "scripts/run_exp012_cpu_official_sequence.py",
        "scripts/eval_exp012_cpu_task.py",
        "scripts/eval_exp012_wikitext103.py",
        "lm_eval",
    )
    processes: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid() and any(marker in command for marker in markers):
            processes.append((pid, command))
    return processes


def assert_no_official_artifacts(output_dir: Path, tasks: Sequence[str]) -> None:
    """Refuse a sequence launch that could overwrite a prior official result."""
    from gibc_llm.official_cpu_evaluation import artifact_path

    for task in tasks:
        artifact = artifact_path(output_dir, task)
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite previous official artifact: {artifact}")


def recover_stale_sequence(
    status_path: Path,
    *,
    output_dir: Path,
    pid_is_alive: Callable[[int], bool] = _pid_is_alive,
    relevant_processes: Callable[[], list[tuple[int, str]]] = _relevant_evaluator_processes,
) -> Path | None:
    """Archive a dead interrupted sequence only after comprehensive liveness checks.

    The suite guard is deliberately excluded: by the time its child sequence
    begins, that guard is the current live owner, not evidence from the stale
    sequence being recovered.
    """
    prior = _read_json(status_path)
    if prior is None or prior.get("state") != "running":
        return None

    stale_records: list[tuple[Path, dict[str, object]]] = [(status_path, prior)]
    for candidate in sorted(status_path.parent.glob("*.status.json")):
        if candidate == status_path or candidate.name == "suite.status.json":
            continue
        record = _read_json(candidate)
        if record is not None and record.get("state") == "running":
            stale_records.append((candidate, record))

    for candidate, record in stale_records:
        for field in ("supervisor_pid", "evaluator_pid"):
            pid = _recorded_pid(record, field, candidate)
            if pid is not None and pid_is_alive(pid):
                raise EvaluationAlreadyRunning(f"Refusing stale-sequence recovery: {candidate} has live recorded PID {pid} in {field}.")
    allowed_processes = {os.getpid()}
    suite_path = status_path.parent / "suite.status.json"
    suite = _read_json(suite_path)
    if suite is not None and suite.get("state") == "running":
        for field in ("supervisor_pid", "evaluator_pid"):
            pid = _recorded_pid(suite, field, suite_path)
            if pid is not None:
                allowed_processes.add(pid)
    for pid, command in relevant_processes():
        if pid not in allowed_processes:
            raise EvaluationAlreadyRunning(f"Refusing stale-sequence recovery: relevant evaluator process is live (PID {pid}: {command}).")

    archive = _new_history_directory(output_dir / "history", "interrupted-by-terminal-closure")
    archived_statuses: list[str] = []
    archived_logs: list[str] = []
    status_destination = archive / "status"
    status_destination.mkdir()
    for candidate, record in stale_records:
        os.replace(candidate, status_destination / candidate.name)
        archived_statuses.append(candidate.name)
        archived_logs.extend(_move_status_logs_to_archive(record, archive))
    from gibc_llm.official_cpu_evaluation import artifact_path

    valid_artifact = any(artifact_path(output_dir, task).is_file() for task in ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"))
    _write_json_atomic(
        archive / "interruption.json",
        {
            "classification": "INTERRUPTED_BY_TERMINAL_CLOSURE",
            "archived_at": _timestamp(),
            "sequence_status": str(status_path),
            "archived_statuses": archived_statuses,
            "archived_logs": archived_logs,
            "valid_result_artifact_produced": valid_artifact,
        },
    )
    return archive


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


def _archive_terminal_status(status_path: Path, prior: dict[str, object] | None) -> None:
    """Keep earlier terminal status evidence before a deliberate retry."""
    if prior is None or prior.get("state") == "running" or not status_path.exists():
        return
    history = status_path.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp_filename(prior.get("terminal_at") or prior.get("started_at") or "unknown")
    target = history / f"{status_path.stem}.{timestamp}.json"
    suffix = 1
    while target.exists():
        target = history / f"{status_path.stem}.{timestamp}.{suffix}.json"
        suffix += 1
    os.replace(status_path, target)


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
            for field in ("supervisor_pid", "evaluator_pid"):
                pid = _recorded_pid(prior, field, status_path)
                if pid is not None and _pid_is_alive(pid):
                    _release_lock(status_path)
                    pid_label = "supervisor PID" if field == "supervisor_pid" else "evaluator PID"
                    raise EvaluationAlreadyRunning(
                        f"Task {task!r} already has a live guarded evaluator ({pid_label} {pid})."
                    )
                if pid is not None and stale_status_pid is None:
                    stale_status_pid = pid
            _archive_interrupted_guarded_attempt(status_path, prior)
            prior = None

    _archive_terminal_status(status_path, prior)

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
