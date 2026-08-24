from gibc_llm.evaluation_output import evaluation_output_record


def test_evaluation_output_persists_protocol_timing_and_unmodified_raw_result() -> None:
    """Breaks if task timing/provenance is only printed or if the raw harness result is rewritten."""
    raw_result = {"results": {"hellaswag": {"acc,none": 0.25}}, "config": {"limit": None}}

    output = evaluation_output_record(
        task="hellaswag",
        checkpoint="artifacts/exp004-full/run/checkpoints/checkpoint-step-9156.pt",
        tokenizer="artifacts/exp001c-full/tokenizer/tokenizer.json",
        tokenizer_sha256="c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14",
        batch_size=16,
        lm_eval_version="0.4.9.1",
        num_fewshot=0,
        wall_seconds=12.5,
        raw_result=raw_result,
    )

    assert output["metadata"] == {
        "task": "hellaswag",
        "checkpoint": "artifacts/exp004-full/run/checkpoints/checkpoint-step-9156.pt",
        "tokenizer": "artifacts/exp001c-full/tokenizer/tokenizer.json",
        "tokenizer_sha256": "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14",
        "batch_size": 16,
        "lm_eval_version": "0.4.9.1",
        "num_fewshot": 0,
        "wall_seconds": 12.5,
    }
    assert output["raw_lm_eval_result"] is raw_result
