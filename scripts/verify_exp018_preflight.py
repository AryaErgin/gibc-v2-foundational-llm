"""No-training EXP-018 QK-Norm and exact EXP-011-prefix data-reuse preflight."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gibc_llm.exp012 import (
    EXP011_MANIFEST_SHA256,
    EXP011_PREDICTION_TOKENS,
    EXP011_STORED_TOKEN_IDS,
    EXP011_STREAM_SHA256,
    verify_exp011_prefix,
)
from gibc_llm.full_run import (
    EXP012_PREDICTION_TOKENS,
    EXP012_STREAM_SHA256,
    load_full_run_artifact,
)
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config, sha256_file

EXP011_TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
EXP012_STORED_TOKEN_IDS = EXP012_PREDICTION_TOKENS + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp018-qk-norm.yaml"))
    parser.add_argument("--control-config", type=Path, default=Path("configs/exp011.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_config(args.config)
    control = load_config(args.control_config)
    if candidate.experiment_id != "EXP-018" or control.experiment_id != "EXP-011":
        raise RuntimeError("EXP-018 preflight requires the frozen EXP-018 candidate and EXP-011 control configs.")
    candidate_model, control_model = asdict(candidate.model), asdict(control.model)
    for key in ("qk_norm", "qk_norm_epsilon"):
        candidate_model.pop(key)
        control_model.pop(key)
    if candidate_model != control_model or candidate.training != control.training or candidate.data != control.data or candidate.mixture != control.mixture:
        raise RuntimeError("EXP-018 differs from EXP-011 outside the preregistered QK-Norm fields.")
    if not candidate.model.qk_norm or candidate.model.qk_norm_epsilon != 1.0e-6:
        raise RuntimeError("EXP-018 QK-Norm flag or epsilon differs from preregistration.")

    model = DecoderOnlyTransformer(candidate.model)
    breakdown = parameter_breakdown(model)
    gain_names = [name for name, parameter in model.named_parameters() if name.endswith("qk_norm_gain") and parameter.requires_grad]
    if breakdown.total != 49_860_489 or len(gain_names) != 9 or any(model.get_parameter(name).item() != 1.0 for name in gain_names):
        raise RuntimeError("EXP-018 QK-Norm parameter-count or gain-initialization invariant failed.")

    artifact_dir = args.artifact_dir
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = manifest.get("packed", {})
    stream_path = artifact_dir / packed.get("train_stream_file", "train-token-stream.uint16")
    if (
        manifest.get("experiment_id") != "EXP-012"
        or packed.get("train_prediction_tokens") != EXP012_PREDICTION_TOKENS
        or packed.get("train_token_count_including_final_target") != EXP012_STORED_TOKEN_IDS
        or packed.get("train_stream_sha256") != EXP012_STREAM_SHA256
        or manifest.get("tokenizer", {}).get("sha256") != EXP011_TOKENIZER_SHA256
    ):
        raise RuntimeError("EXP-018 requires the exact verified EXP-012 2.4B source artifact.")
    # This loader independently hashes the whole EXP-012 stream and checks the immutable EXP-011/006/004 prefix chain.
    artifact = load_full_run_artifact(artifact_dir, candidate)
    observed_prefix_sha256 = verify_exp011_prefix(stream_path, EXP011_STORED_TOKEN_IDS * 2)
    prefix = manifest.get("exp011_prefix", {})
    frozen = manifest.get("frozen_exp011_source", {})
    if (
        observed_prefix_sha256 != EXP011_STREAM_SHA256
        or prefix.get("expected_sha256") != EXP011_STREAM_SHA256
        or prefix.get("observed_sha256") != EXP011_STREAM_SHA256
        or prefix.get("prefix_match") is not True
        or prefix.get("stored_token_ids") != EXP011_STORED_TOKEN_IDS
        or frozen.get("manifest_sha256") != EXP011_MANIFEST_SHA256
        or frozen.get("stream_sha256") != EXP011_STREAM_SHA256
        or frozen.get("stored_token_ids") != EXP011_STORED_TOKEN_IDS
        or frozen.get("prediction_tokens") != EXP011_PREDICTION_TOKENS
        or artifact.train.token_count != EXP011_STORED_TOKEN_IDS
        or len(artifact.train) != EXP011_PREDICTION_TOKENS // candidate.data.context_length
    ):
        raise RuntimeError("EXP-018 exact EXP-011 prefix/token-cursor identity verification failed.")
    token_ids = np.memmap(stream_path, mode="r", dtype=np.uint16, shape=(EXP011_STORED_TOKEN_IDS,))

    print(json.dumps({
        "training_launch": False,
        "official_benchmark_invocation": False,
        "experiment": candidate.experiment_id,
        "config_sha256": sha256_file(args.config),
        "control_config_sha256": sha256_file(args.control_config),
        "parameters": breakdown.total,
        "qk_gain_parameter_names": gain_names,
        "qk_norm_epsilon": candidate.model.qk_norm_epsilon,
        "data_reuse": {
            "artifact_dir": str(artifact_dir),
            "source_experiment": "EXP-012",
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_stream_sha256": packed["train_stream_sha256"],
            "source_stored_token_ids": packed["train_token_count_including_final_target"],
            "prefix_stream_sha256": observed_prefix_sha256,
            "prefix_stored_token_ids": artifact.train.token_count,
            "prefix_prediction_tokens": artifact.train.token_count - 1,
            "prefix_sequences": len(artifact.train),
            "first_token_id": int(token_ids[0]),
            "last_prefix_token_id": int(token_ids[-1]),
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "frozen_exp011_manifest_sha256": frozen["manifest_sha256"],
            "exact_exp011_prefix_identity": True,
            "cursor_semantics": "TokenStreamDataset bounds the read-only memmap to the exact 1,500,020,737-ID prefix; sequence i starts at i*512.",
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
