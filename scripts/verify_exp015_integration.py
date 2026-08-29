"""Short, non-benchmark Schedule-A equivalence and scheduled-resume gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gibc_llm.full_run import load_full_run_artifact
from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import WarmupStableDecay, build_optimizer, load_checkpoint, optimizer_update, save_checkpoint
from gibc_llm.utils import load_config, set_global_seed


def _maximum_difference(left: object, right: object) -> float:
    if isinstance(left, torch.Tensor):
        return float((left.detach().cpu() - right.detach().cpu()).abs().max())
    if isinstance(left, dict):
        return max((_maximum_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        return max((_maximum_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else float("inf")


def _build(config, device):
    set_global_seed(42)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = WarmupStableDecay(optimizer, 6e-4, 6e-5, 100, 9156, 916)
    return model, optimizer, schedule


def _update(model, optimizer, schedule, train, ids, device):
    batches = [train.get_indexed_batch(ids[:32]), train.get_indexed_batch(ids[32:])]
    return optimizer_update(model, optimizer, schedule, batches, device, 1.0)


def main() -> None:
    config = load_config(Path("configs/exp013-wsd.yaml"))
    artifact = load_full_run_artifact(Path("artifacts/exp004-full"), config)
    schedule_ids = np.load("artifacts/exp015-fixed-examples/schedule-a-static.npy", allow_pickle=False)
    expected_hash = "39c509f59489d125904be61e7e3094e0e87af5ee7ead46afe6742cac35185eb2"
    if hashlib.sha256(schedule_ids.tobytes()).hexdigest() != expected_hash:
        raise RuntimeError("Schedule A hash mismatch.")
    device = torch.device("cuda")
    legacy = _build(config, device)
    legacy_repeat = _build(config, device)
    indexed = _build(config, device)
    indexed_repeat = _build(config, device)
    if _maximum_difference(legacy[0].state_dict(), indexed[0].state_dict()) != 0.0:
        raise RuntimeError("Fresh seed-42 model initialization differs.")
    pairs = {"L/L": (legacy, legacy_repeat), "A/A": (indexed, indexed_repeat), "L/A": (legacy, indexed)}
    report = {name: {"loss": [], "model": [], "optimizer": []} for name in pairs}
    for step in range(10):
        ids = list(range(step * 64, (step + 1) * 64))
        scheduled_ids = schedule_ids[step * 64 : (step + 1) * 64].tolist()
        for offset in (0, 32):
            legacy_batch = artifact.train.get_indexed_batch(ids[offset : offset + 32])
            indexed_batch = artifact.train.get_indexed_batch(scheduled_ids[offset : offset + 32])
            if not torch.equal(legacy_batch[0], indexed_batch[0]) or not torch.equal(legacy_batch[1], indexed_batch[1]):
                raise RuntimeError("Schedule-A microbatch differs from legacy contiguous data.")
        results = {"L": _update(*legacy, artifact.train, ids, device), "L2": _update(*legacy_repeat, artifact.train, ids, device), "A": _update(*indexed, artifact.train, scheduled_ids, device), "A2": _update(*indexed_repeat, artifact.train, scheduled_ids, device)}
        for name, ((left_model, left_optimizer, _), (right_model, right_optimizer, _)) in pairs.items():
            left_key, right_key = ("L", "L2") if name == "L/L" else ("A", "A2") if name == "A/A" else ("L", "A")
            report[name]["loss"].append(abs(results[left_key]["loss"] - results[right_key]["loss"]))
            report[name]["model"].append(_maximum_difference(left_model.state_dict(), right_model.state_dict()))
            report[name]["optimizer"].append(_maximum_difference(left_optimizer.state_dict(), right_optimizer.state_dict()))
    print(json.dumps({"microbatches_compared": 30, "optimizer_steps": 10, "cursor": 640, "pairs": {name: {key: [values[0], values[4], values[9]] for key, values in measurements.items()} for name, measurements in report.items()}}, indent=2))


if __name__ == "__main__":
    main()
