"""Sequentially run the authorized frozen EXP-012 CPU official-evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from gibc_llm.evaluation_launch import run_guarded
from gibc_llm.official_cpu_evaluation import AMENDMENT_COMMIT, validate_lm_task_record, validate_wikitext103_record


TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_status(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/exp012.yaml"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('CPU official sequence requires CUDA_VISIBLE_DEVICES="".')
    status_path = args.output_dir / "status" / "sequence.status.json"
    sequence: dict[str, object] = {
        "state": "running",
        "amendment_commit": AMENDMENT_COMMIT,
        "started_at": _timestamp(),
        "terminal_at": None,
        "tasks": [],
    }
    _write_status(status_path, sequence)
    try:
        for task in TASKS:
            artifact = args.output_dir / ("wikitext103.json" if task == "wikitext103" else "lm_eval" / f"{task}.json")
            if artifact.exists():
                raise FileExistsError(f"Refusing to overwrite previous official artifact: {artifact}")
            script = "scripts/eval_exp012_wikitext103.py" if task == "wikitext103" else "scripts/eval_exp012_cpu_task.py"
            command = [
                args.python,
                script,
                "--config",
                str(args.config),
                "--checkpoint",
                str(args.checkpoint),
                "--tokenizer",
                str(args.tokenizer),
                "--output",
                str(artifact),
                "--batch-size",
                "16",
            ]
            if task != "wikitext103":
                command.extend(["--task", task])
            task_status = args.output_dir / "status" / f"{task}.status.json"
            task_stdout = args.output_dir / "logs" / f"{task}.stdout.log"
            task_stderr = args.output_dir / "logs" / f"{task}.stderr.log"
            returncode = run_guarded(
                task=task,
                command=command,
                status_path=task_status,
                stdout_path=task_stdout,
                stderr_path=task_stderr,
                environment={"CUDA_VISIBLE_DEVICES": ""},
            )
            if returncode != 0 or not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"Official task {task} failed or did not write its result artifact.")
            record = json.loads(artifact.read_text(encoding="utf-8"))
            if task == "wikitext103":
                validate_wikitext103_record(record)
            else:
                validate_lm_task_record(record, task)
            completed = {"task": task, "status": "succeeded", "artifact": str(artifact), "artifact_sha256": _sha256(artifact)}
            sequence["tasks"].append(completed)  # type: ignore[index]
            _write_status(status_path, sequence)
    except BaseException as exc:
        sequence["state"] = "stopped"
        sequence["terminal_at"] = _timestamp()
        sequence["error"] = f"{type(exc).__name__}: {exc}"
        _write_status(status_path, sequence)
        raise
    sequence["state"] = "succeeded"
    sequence["terminal_at"] = _timestamp()
    _write_status(status_path, sequence)


if __name__ == "__main__":
    main()
