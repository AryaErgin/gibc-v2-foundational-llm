"""Release tests never perform inference, benchmark requests or dataset access."""
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file
from gibc_llm.inference_release import (
    compare_states, export_tensors, write_manifest, verify_files, strict_cpu_model,
)


def test_exact_export_shared_weights_and_determinism(tmp_path):
    x = torch.tensor([[1.0, -0.0], [2.0, 3.0]])
    state = {"embedding": x, "output": x, "norm": torch.ones(2)}
    a, b = tmp_path/"a.safetensors", tmp_path/"b.safetensors"
    export_tensors(state, a)
    export_tensors(state, b)
    assert a.read_bytes() == b.read_bytes()
    restored = load_file(str(a))
    assert compare_states(state, restored)["tensor_count"] == 3
    assert restored["embedding"].data_ptr() != restored["output"].data_ptr()
    with pytest.raises(FileExistsError):
        export_tensors(state, a)


@pytest.mark.parametrize("mutation", ["name", "shape", "dtype", "value", "signed_zero", "nan"])
def test_tensor_comparison_rejects_drift(mutation):
    old = {"x": torch.tensor([0.0, 1.0])}
    new = {"x": old["x"].clone()}
    if mutation == "name": new = {"y": new["x"]}
    if mutation == "shape": new["x"] = new["x"].reshape(1, 2)
    if mutation == "dtype": new["x"] = new["x"].double()
    if mutation == "value": new["x"][1] = 2
    if mutation == "signed_zero": new["x"][0] = -0.0
    if mutation == "nan": new["x"][0] = float("nan")
    with pytest.raises(ValueError): compare_states(old, new)


def test_manifest_exact_files_hashes_no_overwrite(tmp_path):
    (tmp_path/"payload.txt").write_text("original")
    write_manifest(tmp_path)
    assert verify_files(tmp_path)["file_count"] == 1
    with pytest.raises(FileExistsError): write_manifest(tmp_path)
    (tmp_path/"extra").write_text("unexpected")
    with pytest.raises(ValueError): verify_files(tmp_path)
    (tmp_path/"extra").unlink()
    (tmp_path/"payload.txt").write_text("modified")
    with pytest.raises(ValueError): verify_files(tmp_path)


def test_strict_load_no_forward_and_rng_preserved(monkeypatch):
    from gibc_llm.model import DecoderOnlyTransformer
    from gibc_llm.utils import load_config
    from dataclasses import replace
    config = load_config(Path(__file__).parents[1]/"configs/exp020-final-7p2b-cosine.yaml")
    small = replace(config.model, vocab_size=32, d_model=16, n_layers=1, n_heads=2,
                    head_dim=8, d_ff=32, rotary_dim=8, context_length=16)
    original = DecoderOnlyTransformer(small)
    state = original.state_dict()
    before = torch.random.get_rng_state().clone()
    def forbidden(*args, **kwargs): raise AssertionError("inference is prohibited")
    monkeypatch.setattr(DecoderOnlyTransformer, "forward", forbidden)
    count = sum(p.numel() for p in original.parameters() if p.requires_grad)
    result = strict_cpu_model(small, state, expected_count=count)
    assert result["trainable_parameters"] == count
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises((RuntimeError, ValueError)):
        strict_cpu_model(small, {"missing": torch.ones(1)}, expected_count=count)
