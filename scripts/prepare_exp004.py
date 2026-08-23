"""Materialize the globally deduplicated EXP-004 FineWeb/FineWeb-Edu mixture; does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.exp004 import prepare_exp004
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp004.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--exp001-artifact-dir", type=Path, default=Path("artifacts/exp001c-full"))
    parser.add_argument("--exp003-artifact-dir", type=Path, default=Path("artifacts/exp003-full"))
    args = parser.parse_args()
    manifest = prepare_exp004(load_config(args.config), args.artifact_dir, args.exp001_artifact_dir, args.exp003_artifact_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
