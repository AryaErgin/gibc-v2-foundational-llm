"""Print a reproducible trainable-parameter accounting for any model config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, help="Fail unless the counted total equals this exact value.")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable evidence record.")
    args = parser.parse_args()
    config = load_config(args.config)
    breakdown = parameter_breakdown(DecoderOnlyTransformer(config.model))
    record = {
        "config": str(args.config).replace("\\", "/"),
        "config_sha256": sha256_file(args.config),
        **breakdown.__dict__,
    }
    if args.expected_total is not None:
        record["expected_total"] = args.expected_total
        if breakdown.total != args.expected_total:
            raise SystemExit(f"Parameter invariant failed: expected {args.expected_total:,}, got {breakdown.total:,}.")
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    for name, value in record.items():
        print(f"{name}: {value:,}" if isinstance(value, int) else f"{name}: {value}")


if __name__ == "__main__":
    main()
