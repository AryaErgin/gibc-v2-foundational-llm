"""Unit contracts for inference-only publication artifacts."""

from __future__ import annotations

from collections import OrderedDict
import json

from safetensors.torch import load_file
import torch

from gibc_llm.publication import _model_card, inference_state, save_safetensors, verify_manifest, write_manifest


def test_inference_state_contains_only_cpu_model_tensors() -> None:
    source_state = OrderedDict(
        weight=torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1),
    )

    exported_state = inference_state(source_state)

    assert list(exported_state) == ["weight"]
    exported = exported_state["weight"]
    assert exported.device.type == "cpu"
    assert exported.is_contiguous()
    assert not exported.requires_grad
    assert torch.equal(exported, source_state["weight"])


def test_safetensors_export_round_trips_only_named_tensors(tmp_path) -> None:
    source_state = OrderedDict(weight=torch.arange(6, dtype=torch.float32).reshape(2, 3))
    destination = tmp_path / "model.safetensors"

    save_safetensors(source_state, destination)
    loaded = load_file(destination, device="cpu")

    assert destination.is_file()
    assert set(loaded) == {"weight"}
    assert loaded["weight"].shape == (2, 3)
    assert loaded["weight"].dtype == torch.float32
    assert torch.equal(loaded["weight"], source_state["weight"])


def test_manifest_lists_payload_files_in_sorted_order_and_verifies(tmp_path) -> None:
    (tmp_path / "zeta.txt").write_bytes(b"zeta\n")
    (tmp_path / "alpha.txt").write_bytes(b"alpha\n")

    manifest_path = write_manifest(tmp_path)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["alpha.txt", "manifest.json", "zeta.txt"]
    assert all(len(line.split("  ", 1)[0]) == 64 for line in lines)
    record = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in record["files"]] == ["alpha.txt", "zeta.txt"]
    assert verify_manifest(tmp_path) == 3


def test_model_card_describes_an_evaluated_checkpoint_not_a_final_submission() -> None:
    card = _model_card()

    for required in (
        "decoder-only language model trained entirely from scratch",
        "EXP-012 evaluated research checkpoint",
        "exp012-evaluated-v1",
        "Apache-2.0",
        "validated local publication candidate",
        "Public model publication is deferred",
        "49,860,480 trainable parameters",
        "2,399,993,856 prediction tokens",
        "FineWeb/FineWeb-Edu 2:1 Data Recipe v1",
        "No pretrained initialization, pretrained weights, fine-tuning, or distillation was used.",
        "This checkpoint may later be superseded by a final GIBC model.",
        "WinoGrande accuracy is near chance.",
        "finite contamination screen",
        "MODEL_DIR=/absolute/path/to/gibc-v2-track01-exp012-evaluated-checkpoint",
        "0.28759211312487554",
        "0.36447811447811446",
        "0.6022850924918389",
        "0.5035516969218626",
        "35.93897257521639",
    ):
        assert required in card
