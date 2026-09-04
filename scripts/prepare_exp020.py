"""Materialize the final unique EXP-020 7.2B token stream; does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.exp020 import prepare_exp020
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp020-final-7p2b-cosine.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--exp012-artifact-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True, help="Fresh native-WSL hot scratch root; never /mnt.")
    parser.add_argument("--progress-interval-stored-ids", type=int, default=10_000_000)
    parser.add_argument("--contamination-index-mode", choices=("sqlite", "memory"), default="sqlite")
    parser.add_argument("--recorded-source-commit", required=True, help="Immutable committed EXP-020 builder/config source SHA.")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_exp020(
                load_config(args.config),
                args.artifact_dir,
                args.exp012_artifact_dir,
                args.recorded_source_commit,
                args.scratch_dir,
                args.progress_interval_stored_ids,
                contamination_index_mode=args.contamination_index_mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
