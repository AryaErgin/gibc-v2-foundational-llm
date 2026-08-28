"""Build a local, inference-only package for the evaluated EXP-012 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.publication import build_exp012_publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--official-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = build_exp012_publication(
        source_checkpoint=args.source_checkpoint,
        source_tokenizer=args.source_tokenizer,
        source_config=args.source_config,
        official_provenance=args.official_provenance,
        output_dir=args.output_dir,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
