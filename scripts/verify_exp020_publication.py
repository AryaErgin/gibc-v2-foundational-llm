"""Verify a release without inference, data loading or network access."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from gibc_llm.inference_release import verify_release

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.package), indent=2, sort_keys=True))
