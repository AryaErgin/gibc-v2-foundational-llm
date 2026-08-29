"""Finalize EXP-015 immutable-window diagnostics from a verified replay; never trains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.exp015 import finalize_preflight_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp004-full"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/exp015-fixed-examples"))
    args = parser.parse_args()
    print(json.dumps(finalize_preflight_report(args.artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
