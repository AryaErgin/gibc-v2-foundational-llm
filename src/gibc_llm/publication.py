"""Deterministic, inference-only packaging for the evaluated EXP-012 checkpoint."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from .model import DecoderOnlyTransformer, parameter_breakdown
from .utils import load_config, sha256_file


EXPECTED_CHECKPOINT_SHA256 = "cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c"
EXPECTED_TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
EXPECTED_CONFIG_SHA256 = "d1b7c9e6e940e009c3eb84b6b560dffa166dbf29c22d49a3547c26e7a27ea1c7"
EXPECTED_TRAINABLE_PARAMETERS = 49_860_480
TRAINING_SOURCE_COMMIT = "4b22f8d7a7eacbdd315cbb454a813203ae410c1d"
SOURCE_REPOSITORY = "https://github.com/AryaErgin/gibc-v2-foundational-llm"
MANIFEST_NAME = "SHA256SUMS"
METADATA_MANIFEST_NAME = "manifest.json"
ARTIFACT_VERSION = "exp012-evaluated-v1"
FIXED_INPUT_IDS = ((1, 2, 3, 4), (5, 6, 7, 8))
PROJECT_LICENSE = Path(__file__).resolve().parents[2] / "LICENSE"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def _require_hash(path: Path, expected: str, description: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{description} SHA-256 mismatch: expected {expected}, got {actual}.")
    return actual


def inference_state(model_state: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
    """Return the sole permitted inference payload: named CPU FP32 model tensors."""
    if not model_state:
        raise ValueError("Model state must not be empty.")
    exported: OrderedDict[str, Tensor] = OrderedDict()
    for name, tensor in model_state.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("Inference state must map string parameter names to tensors only.")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Inference state tensor {name!r} must be CPU FP32, got {tensor.dtype}.")
        exported[name] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    return exported


def save_safetensors(model_state: Mapping[str, Tensor], destination: Path) -> None:
    """Serialize only the strict model state in non-pickle safetensors format."""
    save_file(inference_state(model_state), str(destination))


def _validate_tensor_inventory(expected: Mapping[str, Tensor], exported: Mapping[str, Tensor]) -> dict[str, dict[str, Any]]:
    if set(expected) != set(exported):
        raise RuntimeError("Safetensors tensor names differ from the source model state.")
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        source_tensor, exported_tensor = expected[name], exported[name]
        if not isinstance(source_tensor, Tensor) or not isinstance(exported_tensor, Tensor):
            raise RuntimeError(f"Tensor inventory entry {name!r} is not a tensor.")
        if source_tensor.shape != exported_tensor.shape or source_tensor.dtype != exported_tensor.dtype:
            raise RuntimeError(f"Safetensors tensor shape or dtype mismatch for {name!r}.")
        if source_tensor.device.type != "cpu" or exported_tensor.device.type != "cpu":
            raise RuntimeError(f"Safetensors tensor {name!r} is not on CPU.")
        if not torch.isfinite(source_tensor).all() or not torch.isfinite(exported_tensor).all():
            raise RuntimeError(f"Non-finite tensor encountered in {name!r}.")
        inventory[name] = {
            "dtype": str(exported_tensor.dtype),
            "finite": True,
            "numel": exported_tensor.numel(),
            "shape": list(exported_tensor.shape),
        }
    return inventory


def _manifest_entries(directory: Path, excluded_names: set[str]) -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded_names:
            relative = path.relative_to(directory).as_posix()
            entries.append((sha256_file(path), path.stat().st_size, relative))
    return entries


def write_manifest(directory: Path) -> Path:
    """Write standard SHA256SUMS plus a byte-count JSON manifest for the payload."""
    payload_entries = _manifest_entries(directory, {MANIFEST_NAME, METADATA_MANIFEST_NAME})
    _write_json(
        directory / METADATA_MANIFEST_NAME,
        {
            "files": [
                {"path": relative, "sha256": digest, "size_bytes": size}
                for digest, size, relative in payload_entries
            ],
            "schema_version": 1,
        },
    )
    entries = _manifest_entries(directory, {MANIFEST_NAME})
    manifest = directory / MANIFEST_NAME
    manifest.write_text("".join(f"{digest}  {name}\n" for digest, _, name in entries), encoding="utf-8")
    return manifest


def verify_manifest(directory: Path) -> int:
    """Validate the exact payload file set, sizes, and SHA-256 values in ``SHA256SUMS``."""
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        raise RuntimeError(f"Missing manifest: {manifest}")
    metadata_manifest = directory / METADATA_MANIFEST_NAME
    if not metadata_manifest.is_file():
        raise RuntimeError(f"Missing byte-count manifest: {metadata_manifest}")
    metadata_record = json.loads(metadata_manifest.read_text(encoding="utf-8"))
    expected_payload = [
        (entry["sha256"], entry["size_bytes"], entry["path"])
        for entry in metadata_record.get("files", [])
    ]
    actual_payload = _manifest_entries(directory, {MANIFEST_NAME, METADATA_MANIFEST_NAME})
    if expected_payload != actual_payload:
        raise RuntimeError("Byte-count manifest does not match the packaged payload files.")
    expected: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed manifest line: {line!r}")
        digest, relative = parts
        relative_path = Path(relative)
        if len(digest) != 64 or relative_path.is_absolute() or ".." in relative_path.parts or relative == MANIFEST_NAME:
            raise RuntimeError(f"Unsafe manifest entry: {line!r}")
        expected.append((digest, relative))
    if expected != sorted(expected, key=lambda entry: entry[1]) or len({entry[1] for entry in expected}) != len(expected):
        raise RuntimeError("Manifest entries must be unique and sorted by path.")
    actual = [(digest, relative) for digest, _, relative in _manifest_entries(directory, {MANIFEST_NAME})]
    if expected != actual:
        raise RuntimeError("Manifest does not match the packaged payload files.")
    return len(expected)


def _strict_model(config_path: Path, state: Mapping[str, Tensor]) -> DecoderOnlyTransformer:
    config = load_config(config_path)
    model = DecoderOnlyTransformer(config.model).cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("Loaded model parameter count does not match EXP-012.")
    return model


def _validate_official_provenance(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise RuntimeError(f"Official provenance {key} does not match EXP-012.")
    results = record.get("official_results", {})
    expected_results = {
        ("hellaswag", "acc_norm"): 0.28759211312487554,
        ("arc_easy", "acc_norm"): 0.36447811447811446,
        ("piqa", "acc_norm"): 0.6022850924918389,
        ("winogrande", "acc"): 0.5035516969218626,
        ("wikitext103", "perplexity"): 35.93897257521639,
        ("wikitext103", "bits_per_byte"): 1.4083853215598,
    }
    for (task, metric), expected in expected_results.items():
        if results.get(task, {}).get(metric) != expected:
            raise RuntimeError(f"Official provenance {task}.{metric} does not match EXP-012.")
    return record


def _model_card() -> str:
    return f"""---
license: apache-2.0
language:
- en
tags:
- causal-lm
- decoder-only
- from-scratch
- gibc-track-01
---

# GIBC V2 Track 01 — EXP-012 evaluated research checkpoint

This is an inference-only export of the **EXP-012 evaluated research
checkpoint**, a decoder-only language model trained entirely from scratch for
GIBC V2 Track 01. It is the current best evaluated checkpoint in this research
record, not a claim of a permanently frozen or final GIBC submission model.
This checkpoint may later be superseded by a final GIBC model. Future EXP-013+
research remains possible.

**Artifact version:** `{ARTIFACT_VERSION}`

**Distribution status:** This is a validated local publication candidate.
Public model publication is deferred until the final GIBC model is selected.
EXP-012 may later be retained as a historical version if useful.

## Model

- 49,860,480 trainable parameters.
- Decoder-only causal Transformer: vocabulary 8,192; width 640; 9 layers; 20
  attention heads × 32 dimensions; SwiGLU `d_ff=1728`; RoPE; pre-RMSNorm;
  tied input/output embeddings; context length 512.
- Trained from scratch for 2,399,993,856 prediction tokens. No pretrained initialization, pretrained weights, fine-tuning, or distillation was used.

## Training data and controls

The frozen **FineWeb/FineWeb-Edu 2:1 Data Recipe v1** uses
`HuggingFaceFW/fineweb` (`sample-10BT`, revision
`9bb295ddab0e05d785b879661af7260fed5140fc`) and
`HuggingFaceFW/fineweb-edu` (`default`, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`), with global
canonical-content-SHA-256 deduplication. A finite NFKC/casefold/tokenized
normalized 13-gram screen was used against the indexed benchmark sources.
This screen is evidence against detected overlap, not proof of absence of all
lexical, semantic, or unknown-source contamination.

## Official EXP-012 evaluation

The frozen CPU FP32, zero-shot, batch-16, context-512 protocol completed on
2026-08-28 using `lm-eval==0.4.9.1` for the four reasoning tasks and the
separate competition-correct WikiText-103 evaluator:

| Task | Metric | Value |
|---|---:|---:|
| HellaSwag | acc_norm | 0.28759211312487554 |
| ARC-Easy | acc_norm | 0.36447811447811446 |
| PIQA | acc_norm | 0.6022850924918389 |
| WinoGrande | acc | 0.5035516969218626 |
| WikiText-103 held-out | perplexity | 35.93897257521639 |
| WikiText-103 held-out | bits per byte | 1.4083853215598 |

The exact protocol and raw-artifact hashes are documented in
[`experiments/EXP-012-official-evaluation.md`]({SOURCE_REPOSITORY}/blob/main/experiments/EXP-012-official-evaluation.md).
These results are evidence only; they did not select or alter the checkpoint.

## Inference

Clone the source repository, set `MODEL_DIR` to this downloaded package, and
run the explicit local inference command from the source-repository root:

```bash
MODEL_DIR=/absolute/path/to/gibc-v2-track01-exp012-evaluated-checkpoint
python scripts/generate.py "A short prompt" \\
  --config "$MODEL_DIR/exp012.yaml" \\
  --checkpoint "$MODEL_DIR/model.safetensors" \\
  --tokenizer "$MODEL_DIR/tokenizer.json" \\
  --device cpu --max-new-tokens 64 --temperature 0.0
```

`model.safetensors` is a non-pickle safetensors file containing only named,
strict CPU FP32 model tensors. It contains no optimizer, scheduler, RNG,
cursor, metrics, or other training state.

## Provenance and reproducibility

- Source repository: {SOURCE_REPOSITORY}
- Training source commit: [`{TRAINING_SOURCE_COMMIT}`]({SOURCE_REPOSITORY}/commit/{TRAINING_SOURCE_COMMIT})
- Source terminal checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`
- Frozen tokenizer SHA-256: `{EXPECTED_TOKENIZER_SHA256}`
- `provenance.json`, `architecture.json`, `official-evaluation.json`, and
  `equivalence.json` provide machine-readable provenance and validation.
- `SHA256SUMS` records the SHA-256 and byte size of each shipped payload file.

## Limitations and intended use

This small research model is suitable for controlled local experimentation,
not safety-critical or high-stakes use. WinoGrande accuracy is near chance.
The finite contamination screen does not establish complete decontamination.
Training reproduction needs the pinned data revisions and original local data
artifacts; this package is designed for inference reproduction only.

## License, credits, and AI assistance

The original GIBC V2 code and exported model weights are distributed under
Apache-2.0; see `LICENSE` and `LICENSE-NOTICE.md`. This does not
relicense the training datasets, Common Crawl, or underlying third-party web
content. Credits and data-source notes are in `DATA_SOURCES.md`. AI assistance supported specifications,
implementation, tests, audits, and documentation; human review retained
authority over architecture, data, checkpoint selection, benchmarks, and
publication. AI did not provide pretrained weights, synthetic training data,
benchmark answers, or benchmark-driven checkpoint selection.
"""


def _data_sources_note() -> str:
    return """# Data sources and reproducibility notes

## FineWeb/FineWeb-Edu 2:1 Data Recipe v1

- FineWeb: `HuggingFaceFW/fineweb`, config `sample-10BT`, revision
  `9bb295ddab0e05d785b879661af7260fed5140fc`, text field `text`, ODC-By 1.0.
- FineWeb-Edu: `HuggingFaceFW/fineweb-edu`, config `default`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, text field `text`, ODC-By 1.0.
- Target prediction-token contributions: FineWeb 1,599,995,904; FineWeb-Edu
  799,997,952. Total: 2,399,993,856 prediction tokens.
- Global whole-document canonical-content-SHA-256 deduplication was applied.
- The finite contamination screen used normalized NFKC/casefold/tokenized
  13-gram SHA-256 overlap against the indexed benchmark sources. It does not
  prove absence of all contamination.

This package redistributes no training data, benchmark examples, raw evaluator
outputs, or tokenizer-training text. Apache-2.0 covers only this project's
original code and model-weight distribution; it does not purport to relicense
the underlying training corpus, Common Crawl, or third-party web content.
Dataset/Common Crawl/source rights and upstream attribution obligations remain
applicable.
"""


def _license_notice() -> str:
    return """# Distribution and dataset license notice

The original GIBC V2 code and the exported EXP-012 model weights are
distributed under Apache License 2.0; see `LICENSE`.

The model was trained using `HuggingFaceFW/fineweb`, revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, ODC-By 1.0, and
`HuggingFaceFW/fineweb-edu`, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, ODC-By 1.0. Apache-2.0 does not
purport to relicense either dataset, Common Crawl, or the underlying
third-party web content. Dataset/Common Crawl/source rights and attribution
requirements remain applicable.
"""


def build_exp012_publication(
    *,
    source_checkpoint: Path,
    source_tokenizer: Path,
    source_config: Path,
    official_provenance: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build and verify an EXP-012 inference-only package without benchmarks."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite publication directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to reuse interrupted publication staging directory: {staging_dir}")

    source_checkpoint = source_checkpoint.resolve()
    source_tokenizer = source_tokenizer.resolve()
    source_config = source_config.resolve()
    official_provenance = official_provenance.resolve()
    checkpoint_sha256 = _require_hash(source_checkpoint, EXPECTED_CHECKPOINT_SHA256, "Source checkpoint")
    tokenizer_sha256 = _require_hash(source_tokenizer, EXPECTED_TOKENIZER_SHA256, "Source tokenizer")
    config_sha256 = _require_hash(source_config, EXPECTED_CONFIG_SHA256, "Source config")
    official_record = _validate_official_provenance(official_provenance)

    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, Mapping) or not isinstance(source_payload.get("model"), Mapping):
        raise RuntimeError("Source checkpoint must contain a model state dictionary.")
    source_state = source_payload["model"]
    original_model = _strict_model(source_config, source_state)
    exported_state = inference_state(source_state)

    staging_dir.mkdir()
    model_path = staging_dir / "model.safetensors"
    save_safetensors(source_state, model_path)
    shutil.copyfile(source_tokenizer, staging_dir / "tokenizer.json")
    shutil.copyfile(source_config, staging_dir / "exp012.yaml")
    shutil.copyfile(official_provenance, staging_dir / "official-evaluation.json")
    if not PROJECT_LICENSE.is_file():
        raise RuntimeError(f"Missing project Apache-2.0 license: {PROJECT_LICENSE}")
    shutil.copyfile(PROJECT_LICENSE, staging_dir / "LICENSE")
    (staging_dir / "README.md").write_text(_model_card(), encoding="utf-8")
    (staging_dir / "DATA_SOURCES.md").write_text(_data_sources_note(), encoding="utf-8")
    (staging_dir / "LICENSE-NOTICE.md").write_text(_license_notice(), encoding="utf-8")

    exported_loaded = load_file(model_path, device="cpu")
    with safe_open(model_path, framework="pt", device="cpu") as safe_file:
        safetensors_metadata = safe_file.metadata()
    if safetensors_metadata is not None:
        raise RuntimeError("Exported safetensors file must not carry auxiliary metadata.")
    tensor_inventory = _validate_tensor_inventory(exported_state, exported_loaded)
    _write_json(
        staging_dir / "tensor-inventory.json",
        {"schema_version": 1, "tensor_count": len(tensor_inventory), "tensors": tensor_inventory},
    )
    exported_model = _strict_model(staging_dir / "exp012.yaml", exported_loaded)

    input_ids = torch.tensor(FIXED_INPUT_IDS, dtype=torch.long, device="cpu")
    torch.set_num_threads(1)
    with torch.inference_mode():
        original_logits = original_model(input_ids)
        exported_logits = exported_model(input_ids)
    max_abs_difference = float((original_logits - exported_logits).abs().max().item())
    if not torch.allclose(original_logits, exported_logits, rtol=0.0, atol=1.0e-6):
        raise RuntimeError(f"Inference logits mismatch: max_abs_difference={max_abs_difference}")
    input_bytes = json.dumps(FIXED_INPUT_IDS, separators=(",", ":")).encode("utf-8")
    equivalence = {
        "atol": 1.0e-6,
        "device": "cpu",
        "dtype": "float32",
        "input_ids": [list(row) for row in FIXED_INPUT_IDS],
        "input_ids_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "max_abs_logit_difference": max_abs_difference,
        "method": "strict reload of original full-checkpoint and exported safetensors model state into CPU FP32 models",
        "rtol": 0.0,
        "status": "passed",
        "tensor_inventory_sha256": sha256_file(staging_dir / "tensor-inventory.json"),
        "weight_format": "safetensors",
    }
    _write_json(staging_dir / "equivalence.json", equivalence)

    breakdown = parameter_breakdown(exported_model)
    architecture = {
        "architecture": "decoder_only_transformer",
        "config_sha256": config_sha256,
        "context_length": 512,
        "model": load_config(staging_dir / "exp012.yaml").as_dict()["model"],
        "parameter_breakdown": {
            "attention": breakdown.attention,
            "embedding": breakdown.embedding,
            "mlp": breakdown.mlp,
            "norms": breakdown.norms,
            "output_head_additional": breakdown.output_head_additional,
            "total": breakdown.total,
        },
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
    }
    if breakdown.total != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("Exported inference checkpoint parameter recount failed.")
    _write_json(staging_dir / "architecture.json", architecture)

    provenance = {
        "artifact_kind": "inference_only_evaluated_research_checkpoint",
        "artifact_version": ARTIFACT_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source_path": "artifacts/exp012-full/checkpoints/checkpoint-step-73242.pt",
        "exported_model_sha256": sha256_file(model_path),
        "exported_tensor_count": len(exported_loaded),
        "model_weight_filename": "model.safetensors",
        "model_weight_format": "safetensors",
        "official_evaluation": official_record,
        "optimizer_scheduler_rng_or_training_state_included": False,
        "safetensors_metadata": safetensors_metadata,
        "source_checkpoint_non_model_top_level_keys": sorted(str(key) for key in source_payload if key != "model"),
        "source_repository": SOURCE_REPOSITORY,
        "source_repository_training_commit": TRAINING_SOURCE_COMMIT,
        "tensor_inventory_sha256": sha256_file(staging_dir / "tensor-inventory.json"),
        "tokenizer_sha256": tokenizer_sha256,
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
    }
    _write_json(staging_dir / "provenance.json", provenance)
    manifest = write_manifest(staging_dir)
    file_count = verify_manifest(staging_dir)
    shutil.move(str(staging_dir), str(output_dir))
    return {
        "artifact_path": str(output_dir),
        "exported_model_sha256": sha256_file(output_dir / "model.safetensors"),
        "file_count_excluding_manifest": file_count,
        "manifest_sha256": sha256_file(output_dir / MANIFEST_NAME),
        "max_abs_logit_difference": max_abs_difference,
        "parameter_count": breakdown.total,
        "source_checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
