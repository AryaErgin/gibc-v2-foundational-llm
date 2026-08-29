"""Reconstruct EXP-015 source attribution for the immutable EXP-004 stream; never trains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.exp015 import reconstruct_exp004_attribution
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp013-wsd.yaml"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp004-full"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/exp015-fixed-examples"))
    args = parser.parse_args()
    report = reconstruct_exp004_attribution(load_config(args.config), args.artifact_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
