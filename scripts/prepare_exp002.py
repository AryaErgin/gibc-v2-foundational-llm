"""Materialize the prefix-controlled EXP-002 stream; performs no training."""
import argparse, json
from pathlib import Path
from gibc_llm.exp002 import prepare_exp002
from gibc_llm.utils import load_config
parser = argparse.ArgumentParser()
parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp002-full"))
parser.add_argument("--exp001-artifact-dir", type=Path, default=Path("artifacts/exp001c-full"))
args = parser.parse_args()
print(json.dumps(prepare_exp002(load_config("configs/exp002.yaml"), args.artifact_dir, args.exp001_artifact_dir), indent=2, sort_keys=True))
