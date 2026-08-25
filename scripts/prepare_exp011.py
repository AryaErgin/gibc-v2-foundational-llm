"""Materialize the exact prefix-verified EXP-011 1.5B stream; does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.exp011 import prepare_exp011
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp011.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--exp006-artifact-dir", type=Path, default=Path("artifacts/exp006-full"))
    args = parser.parse_args()
    manifest = prepare_exp011(load_config(args.config), args.artifact_dir, args.exp006_artifact_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
