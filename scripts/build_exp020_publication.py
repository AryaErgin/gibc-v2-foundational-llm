"""Build one NEW inference-only directory from the exact trusted local EXP-020 checkpoint."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from gibc_llm.inference_release import (
    CHECKPOINT_SHA, CONFIG_SHA, TOKENIZER_SHA, SOURCE_COMMIT,
    sha256, strict_cpu_model, export_tensors, write_manifest, verify_release,
)
from gibc_llm.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Output must be absent; never overwrite release evidence")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError("Export requires explicitly hidden CUDA and CPU only")
    config_path = ROOT/"configs/exp020-final-7p2b-cosine.yaml"
    for path, expected in ((args.checkpoint, CHECKPOINT_SHA), (args.tokenizer, TOKENIZER_SHA),
                           (config_path, CONFIG_SHA)):
        if sha256(path) != expected:
            raise ValueError("Frozen SHA mismatch: " + str(path))
    if args.checkpoint.stat().st_size != 598447091:
        raise ValueError("Frozen checkpoint size mismatch")
    sources = {}
    for name in ("__init__.py", "model.py", "utils.py", "generation.py"):
        rel = "src/gibc_llm/" + name
        historical = subprocess.check_output(["git", "show", SOURCE_COMMIT + ":" + rel], cwd=ROOT)
        if (ROOT/rel).read_bytes() != historical:
            raise ValueError("Model/loader source drift: " + rel)
        sources[rel] = sha256(ROOT/rel)
    # weights_only=False is limited to this trusted local file AFTER exact SHA/size gates.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run = payload["run_state"]
    if (run["step"], run["tokens"], run["next_sequence_index"]) != (219726, 7199981568, 14062464):
        raise ValueError("Frozen terminal state mismatch")
    state = payload["model"]
    config = load_config(config_path)
    counted = strict_cpu_model(config.model, state)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    # An interrupted build has no complete manifest and must not be published.
    equality = export_tensors(state, out/"model.safetensors")
    for src, dst in ((config_path, "config.yaml"), (args.tokenizer, "tokenizer.json"),
                     (ROOT/"LICENSE", "LICENSE"), (ROOT/"docs/submission/EXP020_MODEL_CARD.md", "README.md"),
                     (ROOT/"docs/submission/LICENSE_AUDIT.md", "THIRD_PARTY_NOTICES.md"),
                     (ROOT/"results/exp020-submission-evidence.json", "training-evaluation-evidence.json"),
                     (ROOT/"results/exp020-evaluation-source-provenance.json", "evaluation-source-provenance.json"),
                     (ROOT/"scripts/exp020_release_generate.py", "generate.py"),
                     (ROOT/"scripts/verify_exp020_publication.py", "verify.py")):
        shutil.copyfile(src, out/dst)
    (out/"gibc_llm").mkdir()
    for name in ("__init__.py", "model.py", "utils.py", "generation.py", "inference_release.py"):
        shutil.copyfile(ROOT/"src/gibc_llm"/name, out/"gibc_llm"/name)
    versions = {n: importlib.metadata.version(n) for n in
                ("torch", "numpy", "PyYAML", "tokenizers", "safetensors")}
    # CPU verification also works in the qualified CUDA-built torch package with CUDA hidden.
    with (out/"requirements.txt").open("x") as f:
        f.write("# Install the appropriate official torch build separately; tested: " + versions["torch"] + "\n")
        for n in ("numpy", "PyYAML", "tokenizers", "safetensors"):
            f.write(n + "==" + versions[n] + "\n")
    tools = {str(p.relative_to(ROOT)): sha256(p) for p in
             (Path(__file__), ROOT/"src/gibc_llm/inference_release.py",
              ROOT/"scripts/verify_exp020_publication.py", ROOT/"scripts/exp020_release_generate.py")}
    provenance = {
        "experiment": "EXP-020", "source_checkpoint_sha256": CHECKPOINT_SHA,
        "source_checkpoint_bytes": 598447091, "released_weights_sha256": sha256(out/"model.safetensors"),
        "config_sha256": CONFIG_SHA, "tokenizer_sha256": TOKENIZER_SHA,
        "selected_step": 219726, "prediction_tokens": 7199981568,
        "source_commit": SOURCE_COMMIT, "training_source_commit": "d88800733846c7a30e2044fa0a20f9e4a448f328",
        "evaluation_source_commit": SOURCE_COMMIT, "lm_eval_recorded_git_hash": "3733279",
        "model_source_files_sha256": sources, "release_tools_sha256": tools,
        "release_tool_status": "new uncommitted packaging tools; identities are file hashes, not a claimed implementation commit",
        "runtime": {"python": sys.version.split()[0], "packages": versions, "device": "cpu"},
        "tensor_equivalence": equality, "parameter_verification": counted,
        "optimizer_scheduler_rng_training_state_included": False,
        "state_dict_tied_aliases": "EXP-020 has one token_embedding.weight key, also used directly for output projection; 83 tensors, 49,860,480 total elements, no separate output-head tensor.",
        "inference_executed": False,
        "selection": "Checkpoint frozen before EXP-020 official benchmark scoring; no subsequent model selection allowed.",
    }
    with (out/"provenance.json").open("x") as f:
        json.dump(provenance, f, indent=2, sort_keys=True); f.write("\n")
    write_manifest(out)
    print(json.dumps(verify_release(out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
