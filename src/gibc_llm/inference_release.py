"""Inference-distribution checks only: no forward pass, training or dataset access."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save
from .model import DecoderOnlyTransformer
from .utils import load_config

PARAMETERS = 49_860_480
CONFIG_SHA = "25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace"
TOKENIZER_SHA = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
CHECKPOINT_SHA = "95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89"
SOURCE_COMMIT = "37332797909df963ca7c77a945cea8752b60d481"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_states(expected, actual):
    if set(expected) != set(actual):
        raise ValueError("Tensor name inventory differs")
    inventory = {}
    for name in sorted(expected):
        a, b = expected[name], actual[name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError("Non-tensor state: " + name)
        if a.shape != b.shape or a.dtype != b.dtype or a.dtype != torch.float32:
            raise ValueError("Shape/dtype mismatch: " + name)
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite state: " + name)
        # Compare raw bits as well as numerical equality (including signed zero).
        if not torch.equal(a, b) or not torch.equal(a.reshape(-1).view(torch.uint8),
                                                   b.reshape(-1).view(torch.uint8)):
            raise ValueError("Tensor value/bit mismatch: " + name)
        inventory[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "elements": a.numel()}
    return {"tensor_count": len(inventory), "bitwise_equal": True, "tensors": inventory}


def export_tensors(state, destination: Path):
    compare_states(state, state)
    # Clone aliases so both historical tied state_dict keys survive; loading restores the tie.
    copies = {key: state[key].detach().cpu().contiguous().clone() for key in sorted(state)}
    encoded = save(copies)
    with destination.open("xb") as handle:
        handle.write(encoded)
    return compare_states(state, load_file(str(destination), device="cpu"))


def strict_cpu_model(config, state, expected_count=PARAMETERS):
    # Constructor draws are isolated, then completely replaced by the strict frozen state.
    with torch.random.fork_rng(devices=[]):
        model = DecoderOnlyTransformer(config).cpu().float()
    model.load_state_dict(state, strict=True)
    model.eval()
    compare_states(state, model.state_dict())
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != expected_count:
        raise ValueError(f"Parameter count {count} != {expected_count}")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 or not torch.isfinite(p).all()
           for p in model.parameters()):
        raise ValueError("Parameters must be finite CPU FP32")
    return {"trainable_parameters": count, "strict_load": True, "cpu_fp32_finite": True,
            "inference_executed": False,
            "parameter_inventory": {n: list(p.shape) for n, p in model.named_parameters()}}


def payload_files(root: Path):
    # Download/client metadata and Python bytecode are not release payloads.
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and not any(x in {".git", ".cache", "__pycache__"} for x in p.relative_to(root).parts)
                  and p.name not in {"manifest.json", "SHA256SUMS"})


def write_manifest(root: Path):
    if (root/"manifest.json").exists() or (root/"SHA256SUMS").exists():
        raise FileExistsError("Manifest already exists")
    entries = {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
               for p in payload_files(root)}
    with (root/"manifest.json").open("x") as f:
        json.dump({"schema": 1, "files": entries}, f, indent=2, sort_keys=True)
        f.write("\n")
    with (root/"SHA256SUMS").open("x") as f:
        for name in sorted([*entries, "manifest.json"]):
            f.write(sha256(root/name) + "  " + name + "\n")


def verify_files(root: Path):
    record = json.loads((root/"manifest.json").read_text())
    entries = record["files"]
    actual = {p.relative_to(root).as_posix() for p in payload_files(root)}
    if set(entries) != actual:
        raise ValueError("Release file inventory differs")
    for name, item in entries.items():
        p = root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe payload path")
        if p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            raise ValueError("Payload hash/size mismatch: " + name)
    expected_sums = "".join(sha256(root/name) + "  " + name + "\n"
                            for name in sorted([*entries, "manifest.json"]))
    if (root/"SHA256SUMS").read_text() != expected_sums:
        raise ValueError("SHA256SUMS differs")
    return {"file_count": len(entries), "payload_bytes": sum(v["bytes"] for v in entries.values()),
            "manifest_sha256": sha256(root/"manifest.json")}


def verify_release(root: Path):
    files = verify_files(root)
    for name, expected in (("config.yaml", CONFIG_SHA), ("tokenizer.json", TOKENIZER_SHA)):
        if sha256(root/name) != expected:
            raise ValueError("Frozen identity differs: " + name)
    config = load_config(root/"config.yaml")
    if (config.model.context_length != 512 or config.model.qk_norm
        or config.training.cautious_weight_decay or config.training.seed != 42
        or config.training.full_schedule_steps != 219726
        or config.training.full_training_tokens != 7199981568):
        raise ValueError("Frozen EXP-020 config differs")
    state = load_file(str(root/"model.safetensors"), device="cpu")
    model = strict_cpu_model(config.model, state)
    provenance = json.loads((root/"provenance.json").read_text())
    if (provenance["source_checkpoint_sha256"] != CHECKPOINT_SHA
        or provenance["source_checkpoint_bytes"] != 598447091
        or provenance["selected_step"] != 219726
        or provenance["prediction_tokens"] != 7199981568
        or provenance["source_commit"] != SOURCE_COMMIT
        or provenance["released_weights_sha256"] != sha256(root/"model.safetensors")):
        raise ValueError("Frozen release provenance differs")
    return {**files, **model, "released_weights_sha256": sha256(root/"model.safetensors")}
