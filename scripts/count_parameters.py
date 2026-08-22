"""Print the exact EXP-001 trainable parameter accounting."""

from __future__ import annotations

import argparse
from pathlib import Path

from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seed(config.training.seed)
    breakdown = parameter_breakdown(DecoderOnlyTransformer(config.model))
    for name, value in breakdown.__dict__.items():
        print(f"{name}: {value:,}")
    if breakdown.total != 8_392_960:
        raise SystemExit("EXP-001 parameter invariant failed.")


if __name__ == "__main__":
    main()
