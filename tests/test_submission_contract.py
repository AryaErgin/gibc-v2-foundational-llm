"""Judge-facing static and non-benchmark command contracts for Track 01."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exp012_parameter_command_is_explicit_and_verifiable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/count_parameters.py",
            "--config",
            "configs/exp012.yaml",
            "--expected-total",
            "49860480",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["total"] == 49_860_480
    assert record["config"] == "configs/exp012.yaml"


def test_generate_requires_explicit_final_model_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate.py", "judge prompt"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--config" in completed.stderr
    assert "--checkpoint" in completed.stderr
    assert "--tokenizer" in completed.stderr


def test_compact_official_provenance_record_is_safe_and_exact() -> None:
    record = json.loads((ROOT / "results/exp012-official-provenance.json").read_text(encoding="utf-8"))

    assert record["checkpoint_sha256"] == "cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c"
    assert record["trainable_parameters"] == 49_860_480
    assert record["official_results"]["hellaswag"]["acc_norm"] == 0.28759211312487554
    assert record["official_results"]["arc_easy"]["acc_norm"] == 0.36447811447811446
    assert record["official_results"]["piqa"]["acc_norm"] == 0.6022850924918389
    assert record["official_results"]["winogrande"]["acc"] == 0.5035516969218626
    assert record["official_results"]["wikitext103"]["perplexity"] == 35.93897257521639


def test_checked_in_exp012_parameter_evidence_matches_the_final_total() -> None:
    record = json.loads((ROOT / "results/exp012-parameter-count.json").read_text(encoding="utf-8"))

    assert record["config"] == "configs/exp012.yaml"
    assert record["total"] == 49_860_480
    assert record["expected_total"] == 49_860_480


def test_readme_is_finalized_for_judges() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Official benchmarks have not run" not in readme
    for required in (
        "0.28759211312487554",
        "0.36447811447811446",
        "0.6022850924918389",
        "0.5035516969218626",
        "35.93897257521639",
        "AI assistance",
        "count_parameters.py",
        "generate.py",
    ):
        assert required in readme
