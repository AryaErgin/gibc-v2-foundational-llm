"""No-training EXP-019 CWD and EXP-011-prefix preflight."""

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
from gibc_llm.full_run import EXP012_PREDICTION_TOKENS, EXP012_STREAM_SHA256, load_full_run_artifact
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config, sha256_file

TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp019-cwd.yaml"))
    parser.add_argument("--control-config", type=Path, default=Path("configs/exp011.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate, control = load_config(args.config), load_config(args.control_config)
    if candidate.experiment_id != "EXP-019" or control.experiment_id != "EXP-011":
        raise RuntimeError("EXP-019 preflight requires EXP-019 candidate and EXP-011 control configs.")
    candidate_training, control_training = asdict(candidate.training), asdict(control.training)
    candidate_training.pop("cautious_weight_decay")
    control_training.pop("cautious_weight_decay")
    if (
        candidate.model != control.model
        or candidate_training != control_training
        or candidate.data != control.data
        or candidate.mixture != control.mixture
        or not candidate.training.cautious_weight_decay
        or candidate.model.qk_norm
    ):
        raise RuntimeError("EXP-019 must differ from EXP-011 only by CWD enabled with QK-Norm off.")

    model = DecoderOnlyTransformer(candidate.model)
    if parameter_breakdown(model).total != 49_860_480:
        raise RuntimeError("EXP-019 parameter-count invariant failed.")

    manifest_path = args.artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = manifest.get("packed", {})
    stream_path = args.artifact_dir / packed.get("train_stream_file", "train-token-stream.uint16")
    expected_stored = EXP012_PREDICTION_TOKENS + 1
    if (
        manifest.get("experiment_id") != "EXP-012"
        or packed.get("train_prediction_tokens") != EXP012_PREDICTION_TOKENS
        or packed.get("train_token_count_including_final_target") != expected_stored
        or packed.get("train_stream_sha256") != EXP012_STREAM_SHA256
        or manifest.get("tokenizer", {}).get("sha256") != TOKENIZER_SHA256
    ):
        raise RuntimeError("EXP-019 requires the exact verified EXP-012 2.4B source artifact.")
    artifact = load_full_run_artifact(args.artifact_dir, candidate)
    observed_prefix = verify_exp011_prefix(stream_path, EXP011_STORED_TOKEN_IDS * 2)
    prefix, frozen = manifest.get("exp011_prefix", {}), manifest.get("frozen_exp011_source", {})
    if (
        observed_prefix != EXP011_STREAM_SHA256
        or prefix.get("expected_sha256") != EXP011_STREAM_SHA256
        or prefix.get("observed_sha256") != EXP011_STREAM_SHA256
        or prefix.get("prefix_match") is not True
        or prefix.get("stored_token_ids") != EXP011_STORED_TOKEN_IDS
        or frozen.get("manifest_sha256") != EXP011_MANIFEST_SHA256
        or frozen.get("stream_sha256") != EXP011_STREAM_SHA256
        or frozen.get("stored_token_ids") != EXP011_STORED_TOKEN_IDS
        or frozen.get("prediction_tokens") != EXP011_PREDICTION_TOKENS
        or artifact.train.token_count != EXP011_STORED_TOKEN_IDS
        or len(artifact.train) != EXP011_PREDICTION_TOKENS // 512
    ):
        raise RuntimeError("EXP-019 exact EXP-011 prefix/token-cursor verification failed.")
    ids = np.memmap(stream_path, mode="r", dtype=np.uint16, shape=(EXP011_STORED_TOKEN_IDS,))
    print(json.dumps({
        "training_launch": False,
        "official_benchmark_invocation": False,
        "experiment": "EXP-019",
        "config_sha256": sha256_file(args.config),
        "parameters": 49_860_480,
        "qk_norm": False,
        "cautious_weight_decay": {
            "enabled": True,
            "algorithm": "x_next = x - lr * (u + weight_decay * I(u*x >= 0) * x), entrywise; u is the Adam adaptive update; mask reads pre-update x.",
            "source": "Chen et al., Cautious Weight Decay, arXiv:2510.12402v2 / ICLR 2026, Algorithm 1",
            "nominal_weight_decay": candidate.training.weight_decay,
        },
        "data_reuse": {
            "source_path": str(stream_path),
            "source_stream_sha256": packed["train_stream_sha256"],
            "source_stored_ids": packed["train_token_count_including_final_target"],
            "prefix_sha256": observed_prefix,
            "prefix_stored_ids": artifact.train.token_count,
            "prefix_prediction_tokens": artifact.train.token_count - 1,
            "prefix_sequences": len(artifact.train),
            "first_token_id": int(ids[0]),
            "last_prefix_token_id": int(ids[-1]),
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "source_manifest_sha256": sha256_file(manifest_path),
            "frozen_exp011_manifest_sha256": frozen["manifest_sha256"],
            "cursor_semantics": "read-only bounded TokenStreamDataset; sequence i begins at i*512",
        },
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
