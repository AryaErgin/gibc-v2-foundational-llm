from __future__ import annotations

import json
from pathlib import Path

import pytest

from gibc_llm.exp020_official_evaluation import (
    EXP020_OFFICIAL_TASKS,
    EXPECTED_RUNTIME,
    assert_fresh_official_output,
    assert_no_conflicting_evaluator,
    validate_runtime_provenance,
)


def _runtime_record() -> dict[str, object]:
    return {
        "python_version": EXPECTED_RUNTIME["python_version"],
        "torch_version": EXPECTED_RUNTIME["torch_version"],
        "datasets_version": EXPECTED_RUNTIME["datasets_version"],
        "lm_eval_version": EXPECTED_RUNTIME["lm_eval_version"],
        "pyarrow_version": EXPECTED_RUNTIME["pyarrow_version"],
        "sacrebleu_version": EXPECTED_RUNTIME["sacrebleu_version"],
        "cuda_visible_devices": "",
        "torch_cuda_available": False,
        "task_metric_audit": {
            "hellaswag": ["acc", "acc_norm"],
            "arc_easy": ["acc", "acc_norm"],
            "piqa": ["acc", "acc_norm"],
            "winogrande": ["acc"],
        },
    }


def test_runtime_provenance_requires_exact_qualified_versions(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(_runtime_record()), encoding="utf-8")
    validate_runtime_provenance(path)
    incompatible = _runtime_record()
    incompatible["sacrebleu_version"] = "1.5.1"
    path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sacrebleu_version"):
        validate_runtime_provenance(path)


def test_official_output_refuses_any_existing_result_or_status(tmp_path: Path) -> None:
    assert_fresh_official_output(tmp_path)
    artifact = tmp_path / "lm_eval" / "hellaswag.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing"):
        assert_fresh_official_output(tmp_path)
    assert EXP020_OFFICIAL_TASKS == ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103")


def test_duplicate_evaluator_is_rejected_before_output_mutation() -> None:
    with pytest.raises(RuntimeError, match="Conflicting"):
        assert_no_conflicting_evaluator(["999 scripts/eval_exp020_cpu_task.py --task piqa"])


def test_known_outer_supervisor_can_be_explicitly_allowed() -> None:
    assert_no_conflicting_evaluator(
        ["999 scripts/run_exp020_official_sequence.py"],
        allowed_pids={999},
    )


def test_preflight_source_cannot_invoke_or_materialize_benchmarks() -> None:
    source = Path("scripts/preflight_exp020_official_evaluation.py").read_text(encoding="utf-8")
    assert ".simple_evaluate(" not in source
    assert "load_dataset" not in source
    assert "import lm_eval" not in source
    assert "import datasets" not in source
