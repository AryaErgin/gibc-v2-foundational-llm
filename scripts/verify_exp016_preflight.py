"""Read-only EXP-016 Magma integrity preflight; never launches training or benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gibc_llm.full_run import load_full_run_artifact
from gibc_llm.magma import MagmaSettings, magma_blocks, masked_parameter_count
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import WarmupStableDecay, build_optimizer
from gibc_llm.utils import load_config


SCHEDULE_A_SHA256 = "39c509f59489d125904be61e7e3094e0e87af5ee7ead46afe6742cac35185eb2"
STREAM_SHA256 = "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()

    control = load_config(Path("configs/exp016-control.yaml"))
    treatment = load_config(Path("configs/exp016-magma.yaml"))
    if control.magma is not None or treatment.magma is None:
        raise RuntimeError("EXP-016 control/treatment configuration separation failed.")
    settings = MagmaSettings(
        treatment.magma.survival_probability,
        treatment.magma.tau,
        treatment.magma.smoothing,
        treatment.magma.rng_seed,
    )
    if settings != MagmaSettings(0.5, 2.0, 0.9, 42):
        raise RuntimeError("EXP-016 Magma settings differ from preregistration.")
    artifact = load_full_run_artifact(args.artifact_dir, treatment)
    if artifact.manifest["packed"]["train_stream_sha256"] != STREAM_SHA256:
        raise RuntimeError("EXP-016 stream hash differs from frozen EXP-004 stream.")
    indices = np.load(args.schedule, allow_pickle=False)
    observed_schedule_sha = hashlib.sha256(indices.tobytes()).hexdigest()
    if observed_schedule_sha != SCHEDULE_A_SHA256 or indices.dtype != np.uint32 or indices.shape != (len(artifact.train),):
        raise RuntimeError("EXP-016 Schedule A hash/type/shape mismatch.")
    if not np.array_equal(np.sort(indices), np.arange(len(artifact.train), dtype=np.uint32)):
        raise RuntimeError("EXP-016 Schedule A is not an exact immutable-window permutation.")
    model = DecoderOnlyTransformer(treatment.model)
    blocks = magma_blocks(model)
    if parameter_breakdown(model).total != 49_860_480 or len(blocks) != 63 or masked_parameter_count(blocks) != 44_605_440:
        raise RuntimeError("EXP-016 parameter or Magma block mapping invariant failed.")
    optimizer = build_optimizer(model, treatment.training.peak_learning_rate, treatment.training.weight_decay, (treatment.training.beta1, treatment.training.beta2), treatment.training.eps, fused=False)
    wsd = WarmupStableDecay(optimizer, 6e-4, 6e-5, 100, 9156, 916)
    if not (wsd.lr_at_step(100) == 6e-4 and wsd.lr_at_step(8240) == 6e-4 and wsd.lr_at_step(8241) < 6e-4 and wsd.lr_at_step(9156) == 6e-5):
        raise RuntimeError("EXP-016 WSD invariant failed.")
    print(json.dumps({
        "training_launch": False,
        "benchmark_invocation": False,
        "parameters": 49_860_480,
        "masked_blocks": len(blocks),
        "masked_parameters": masked_parameter_count(blocks),
        "excluded_parameters": 5_255_040,
        "magma": {"p": settings.survival_probability, "tau": settings.tau, "smoothing": settings.smoothing, "rng_seed": settings.rng_seed},
        "schedule_sha256": observed_schedule_sha,
        "stream_sha256": artifact.manifest["packed"]["train_stream_sha256"],
        "validation_hashes": {
            "general_inputs": artifact.manifest["general_validation"]["inputs_sha256"],
            "general_targets": artifact.manifest["general_validation"]["targets_sha256"],
            "edu_inputs": artifact.manifest["edu_validation"]["inputs_sha256"],
            "edu_targets": artifact.manifest["edu_validation"]["targets_sha256"],
        },
        "wsd": {"warmup_through": 100, "stable_through": 8240, "cooldown": [8241, 9156], "final_lr": wsd.lr_at_step(9156)},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
