"""Prepare bounded, ignored FineWeb artifacts for EXP-001A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.data import prepare_exp001
from gibc_llm.utils import load_config, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp001a"))
    parser.add_argument("--full-data", action="store_true", help="Prepare the exact 100,007,936-token stream; this does not train.")
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seed(config.training.seed)
    print(json.dumps(prepare_exp001(config, args.artifact_dir, full_run=args.full_data), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
