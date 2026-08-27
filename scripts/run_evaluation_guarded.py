"""Start one official-evaluation command under the durable launch guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gibc_llm.evaluation_launch import EvaluationAlreadyRunning, run_guarded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --; it is executed without a shell.")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide an evaluator command after --")
    try:
        returncode = run_guarded(
            task=args.task,
            command=command,
            status_path=args.status,
            stdout_path=args.stdout,
            stderr_path=args.stderr,
        )
    except EvaluationAlreadyRunning as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
