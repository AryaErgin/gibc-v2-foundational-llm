"""Materialize the frozen-tokenizer EXP-003 FineWeb-Edu artifact without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.exp003 import prepare_exp003
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp003.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frozen-artifact-dir", type=Path, default=Path("artifacts/exp001c-full"))
    parser.add_argument("--validation-only", action="store_true", help="Prepare only held-out edu validation; this is not training-authorized.")
    args = parser.parse_args()
    manifest = prepare_exp003(
        load_config(args.config), args.artifact_dir, args.frozen_artifact_dir, validation_only=args.validation_only
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
