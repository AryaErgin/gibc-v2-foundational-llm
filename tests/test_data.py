import hashlib

import torch

from gibc_llm.data import (
    NgramContaminationFilter,
    TokenStreamDataset,
    assign_split,
    make_packed_examples,
    pack_documents,
    stable_document_id,
    tensor_sha256,
    write_token_stream,
)


def test_sha256_split_is_repeatable_and_has_a_stable_validation_bucket() -> None:
    """Breaks if split assignment depends on dataset iteration order or process RNG."""
    document_id = stable_document_id("A document with normalized\r\nline endings.")

    first = assign_split(document_id, seed=42, modulus=10_000, validation_cutoff=200)
    second = assign_split(document_id, seed=42, modulus=10_000, validation_cutoff=200)

    assert first == second
    expected_bucket = int(hashlib.sha256(f"42:{document_id}".encode("utf-8")).hexdigest(), 16) % 10_000
    assert first == ("validation" if expected_bucket < 200 else "train")


def test_packing_appends_eod_and_advances_by_512_predictions() -> None:
    """Breaks if document boundaries disappear or a packed example supplies only 511 targets."""
    stream = pack_documents([[10, 11], [20, 21]], eod_id=99)
    assert stream.tolist() == [10, 11, 99, 20, 21, 99]

    long_stream = torch.arange(1_025, dtype=torch.long)
    inputs, targets = make_packed_examples(long_stream, context_length=512)
    assert inputs.shape == targets.shape == (2, 512)
    assert torch.equal(inputs[0, :3], torch.tensor([0, 1, 2]))
    assert torch.equal(targets[0, :3], torch.tensor([1, 2, 3]))
    assert inputs[1, 0].item() == 512
    assert targets[1, -1].item() == 1024
    assert tensor_sha256(inputs) == tensor_sha256(inputs.clone())


def test_normalized_ngram_filter_rejects_overlap_without_exposing_text() -> None:
    """Breaks if a normalized benchmark 13-gram overlap is allowed into training."""
    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    contaminated = "prefix one two three four five six seven eight nine ten eleven twelve thirteen suffix"
    clean = "orchards quietly measure copper meteor trails beneath glass lanterns each evening"
    screening = NgramContaminationFilter.from_texts([benchmark], ngram_size=13)

    contaminated_result = screening.screen(contaminated)
    clean_result = screening.screen(clean)

    assert contaminated_result.rejected
    assert contaminated_result.overlap_count >= 1
    assert not clean_result.rejected
    assert benchmark not in str(contaminated_result.as_dict())
    assert contaminated not in str(contaminated_result.as_dict())


def test_uint16_token_stream_matches_packed_input_target_reference(tmp_path) -> None:
    """Breaks if memmapped sequence views change the approved 513-token shift semantics."""
    stream = torch.arange(1_537, dtype=torch.long) % 8192
    reference_inputs, reference_targets = make_packed_examples(stream, context_length=512)
    dataset = write_token_stream(tmp_path / "tokens.uint16", stream.tolist(), token_count=1_537, context_length=512)

    assert isinstance(dataset, TokenStreamDataset)
    assert dataset.storage_dtype == "uint16"
    assert len(dataset) == reference_inputs.shape[0]
    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        assert torch.equal(inputs, reference_inputs[index])
        assert torch.equal(targets, reference_targets[index])


def test_uint16_contiguous_batch_matches_individual_shifted_views(tmp_path) -> None:
    """Breaks if the full-run fast path changes token order or target shifting."""
    stream = torch.arange(3_073, dtype=torch.long) % 8192
    dataset = write_token_stream(tmp_path / "tokens.uint16", stream.tolist(), token_count=3_073, context_length=512)

    inputs, targets = dataset.get_contiguous_batch(1, 4)
    individual = [dataset[index] for index in range(1, 5)]

    assert torch.equal(inputs, torch.stack([item[0] for item in individual]))
    assert torch.equal(targets, torch.stack([item[1] for item in individual]))
