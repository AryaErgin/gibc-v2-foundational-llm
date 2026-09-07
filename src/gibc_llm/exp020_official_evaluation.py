"""Frozen, benchmark-free gates and future-only runners for EXP-020 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


EXP020_OFFICIAL_TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103")
EXPECTED_CHECKPOINT_SHA256 = "95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89"
EXPECTED_CHECKPOINT_BYTES = 598_447_091
EXPECTED_CONFIG_SHA256 = "25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace"
EXPECTED_TOKENIZER_SHA256 = "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14"
EXPECTED_SUMMARY_SHA256 = "31c795afce30b7060b58bd8aa5c8ff14aad98ccb79302c450c8f75d48a995d8a"
EXPECTED_PARAMS = 49_860_480
EXPECTED_SELECTED_STEP = 219_726
EXPECTED_PREDICTION_TOKENS = 7_199_981_568
EXPECTED_SOURCE_COMMIT = "d88800733846c7a30e2044fa0a20f9e4a448f328"
EXPECTED_RUNTIME = {
    "python_version": "3.11.9",
    "torch_version": "2.13.0+cu132",
    "datasets_version": "3.5.1",
    "lm_eval_version": "0.4.9.1",
    "pyarrow_version": "25.0.1",
    "sacrebleu_version": "2.6.0",
}
EXPECTED_TASK_METRICS = {
    "hellaswag": ("acc", "acc_norm"),
    "arc_easy": ("acc", "acc_norm"),
    "piqa": ("acc", "acc_norm"),
    "winogrande": ("acc",),
}
EXPECTED_LM_METRICS = {
    "hellaswag": ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"),
    "arc_easy": ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"),
    "piqa": ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"),
    "winogrande": ("acc,none", "acc_stderr,none"),
}
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(output_dir: Path, task: str) -> Path:
    if task == "wikitext103":
        return output_dir / "wikitext103.json"
    if task in EXPECTED_LM_METRICS:
        return output_dir / "lm_eval" / f"{task}.json"
    raise ValueError(f"Unsupported EXP-020 official task: {task!r}")


def assert_fresh_official_output(output_dir: Path) -> None:
    """Refuse any output location that could conceal prior EXP-020 reporting."""
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"Refusing official evaluation output path that is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite or mix existing EXP-020 official artifacts: {output_dir}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def validate_runtime_provenance(path: Path) -> dict[str, Any]:
    """Hard-gate the separately frozen CPU-only environment and metric audit."""
    record = _read_json(path, "EXP-020 runtime provenance")
    versions = record.get("versions")
    if not isinstance(versions, dict):
        # The compact structure exists solely for unit tests; production must
        # retain the richer frozen provenance record.
        versions = {
            "python": record.get("python_version"),
            "torch": record.get("torch_version"),
            "datasets": record.get("datasets_version"),
            "lm-eval": record.get("lm_eval_version"),
            "pyarrow": record.get("pyarrow_version"),
            "sacrebleu": record.get("sacrebleu_version"),
        }
    names = {
        "python": "python_version",
        "torch": "torch_version",
        "datasets": "datasets_version",
        "lm-eval": "lm_eval_version",
        "pyarrow": "pyarrow_version",
        "sacrebleu": "sacrebleu_version",
    }
    for provenance_name, expected_name in names.items():
        if versions.get(provenance_name) != EXPECTED_RUNTIME[expected_name]:
            raise RuntimeError(
                f"EXP-020 runtime provenance mismatch for {expected_name}: "
                f"expected {EXPECTED_RUNTIME[expected_name]!r}, got {versions.get(provenance_name)!r}."
            )
    if record.get("cuda_visible_devices") != "" or record.get("torch_cuda_available") is not False:
        raise RuntimeError("EXP-020 runtime provenance does not prove CPU-only CUDA isolation.")
    task_audit = record.get("lm_eval_tasks") or record.get("task_metric_audit")
    if not isinstance(task_audit, dict):
        raise RuntimeError("EXP-020 runtime provenance lacks lm-eval task metric audit.")
    for task, expected_metrics in EXPECTED_TASK_METRICS.items():
        audit = task_audit.get(task)
        metrics = audit.get("metrics") if isinstance(audit, dict) else audit
        if tuple(metrics or ()) != expected_metrics:
            raise RuntimeError(f"EXP-020 task metric audit mismatch for {task}: {metrics!r}.")
        if isinstance(audit, dict) and audit.get("sacrebleu_derived_metric") is not False:
            raise RuntimeError(f"EXP-020 task metric audit does not rule out sacrebleu for {task}.")
    return record


def assert_no_conflicting_evaluator(
    process_lines: Iterable[str] | None = None,
    *,
    allowed_pids: set[int] | None = None,
) -> None:
    """Reject a second EXP-020 evaluator before it can touch result paths."""
    if process_lines is None:
        completed = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
        process_lines = completed.stdout.splitlines()
    markers = (
        "run_exp020_official_sequence.py",
        "eval_exp020_cpu_task.py",
        "eval_exp020_wikitext103.py",
    )
    allowed = {str(os.getpid())}
    if allowed_pids is not None:
        allowed.update(str(pid) for pid in allowed_pids)
    for line in process_lines:
        if line.split(None, 1)[:1] and line.split(None, 1)[0] in allowed:
            continue
        if any(marker in line for marker in markers):
            raise RuntimeError(f"Conflicting live EXP-020 official evaluator detected: {line.strip()}")


def _require_payload_state(payload: dict[str, Any]) -> None:
    run_state = payload.get("run_state")
    schedule = payload.get("schedule")
    if not isinstance(run_state, dict) or not isinstance(schedule, dict):
        raise RuntimeError("Selected checkpoint lacks durable run/schedule state.")
    if run_state.get("step") != EXPECTED_SELECTED_STEP or schedule.get("step_count") != EXPECTED_SELECTED_STEP:
        raise RuntimeError("Selected checkpoint step does not equal the frozen terminal EXP-020 step.")
    if run_state.get("tokens") != EXPECTED_PREDICTION_TOKENS:
        raise RuntimeError("Selected checkpoint prediction-token provenance mismatch.")


def load_exp020_cpu_model(args: Any):
    """Load only the frozen terminal checkpoint after all static CPU gates pass.

    This function intentionally imports neither datasets nor lm-eval. Calling it
    is a benchmark-free preflight; task materialization occurs only in the two
    dedicated future execution scripts.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('EXP-020 official evaluation requires CUDA_VISIBLE_DEVICES="" before PyTorch import.')
    if f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" != EXPECTED_RUNTIME["python_version"]:
        raise RuntimeError("Python version differs from the frozen EXP-020 evaluation environment.")
    from importlib.metadata import version

    runtime_packages = {"torch": "torch_version", "datasets": "datasets_version", "lm-eval": "lm_eval_version", "pyarrow": "pyarrow_version", "sacrebleu": "sacrebleu_version"}
    for package, expected_key in runtime_packages.items():
        installed = version(package)
        if installed != EXPECTED_RUNTIME[expected_key]:
            raise RuntimeError(f"Qualified runtime mismatch for {package}: expected {EXPECTED_RUNTIME[expected_key]!r}, got {installed!r}.")
    import torch

    if torch.__version__ != EXPECTED_RUNTIME["torch_version"] or torch.version.cuda != "13.2" or torch.cuda.is_available():
        raise RuntimeError("EXP-020 official evaluation CPU runtime is not the frozen torch/CUDA-isolated environment.")
    from gibc_llm.evaluation import CustomCausalLM
    from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
    from gibc_llm.official_cpu_evaluation import enforce_cpu_isolation
    from gibc_llm.tokenizer import load_tokenizer
    from gibc_llm.utils import load_config

    if args.checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError("Selected EXP-020 checkpoint byte size mismatch.")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Selected EXP-020 checkpoint SHA-256 mismatch.")
    if sha256_file(args.config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("EXP-020 config SHA-256 mismatch.")
    if sha256_file(args.tokenizer) != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("Frozen tokenizer SHA-256 mismatch.")
    if sha256_file(args.summary) != EXPECTED_SUMMARY_SHA256:
        raise RuntimeError("EXP-020 terminal summary SHA-256 mismatch.")
    config = load_config(args.config)
    if (
        config.experiment_id != "EXP-020"
        or config.model.context_length != 512
        or config.model.qk_norm
        or config.training.cautious_weight_decay
        or config.training.optimizer != "adamw"
        or config.training.schedule != "cosine_decay"
        or config.training.full_schedule_steps != EXPECTED_SELECTED_STEP
        or config.training.full_training_tokens != EXPECTED_PREDICTION_TOKENS
    ):
        raise RuntimeError("Frozen EXP-020 scientific configuration contract mismatch.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("Selected checkpoint lacks a strict model state_dict.")
    _require_payload_state(payload)
    model = DecoderOnlyTransformer(config.model).to(torch.device("cpu"))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if parameter_breakdown(model).total != EXPECTED_PARAMS:
        raise RuntimeError("Selected EXP-020 checkpoint parameter-count mismatch.")
    enforce_cpu_isolation(torch, model)
    tokenizer = load_tokenizer(args.tokenizer)
    adapter = CustomCausalLM(model, tokenizer, torch.device("cpu"), batch_size=int(args.batch_size))
    with torch.no_grad():
        direct_logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))[0, -1]
    if not torch.isfinite(direct_logits).all():
        raise FloatingPointError("EXP-020 CPU pre-evaluation direct logits are non-finite.")
    direct_score = float(torch.log_softmax(direct_logits, dim=-1)[4])
    adapter_score, _ = adapter._loglikelihood_tokens([(None, [1, 2, 3], [4])])[0]
    if abs(direct_score - adapter_score) > 1e-6:
        raise RuntimeError("EXP-020 adapter/direct-logit likelihood equivalence gate failed.")
    return torch, config, payload, model, adapter


def atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"EXP-020 official result has non-finite {label}.")
    return float(value)


def validate_lm_task_record(record: dict[str, Any], task: str) -> None:
    if task not in EXPECTED_LM_METRICS:
        raise ValueError(f"Unsupported EXP-020 lm-eval task: {task!r}")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("EXP-020 official result lacks metadata.")
    expected = {
        "task": task,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "trainable_parameters": EXPECTED_PARAMS,
        "selected_step": EXPECTED_SELECTED_STEP,
        "prediction_tokens": EXPECTED_PREDICTION_TOKENS,
        "device": "cpu",
        "precision": "fp32",
        "cuda_available": False,
        "cuda_visible_devices": "",
        "lm_eval_version": EXPECTED_RUNTIME["lm_eval_version"],
        "datasets_version": EXPECTED_RUNTIME["datasets_version"],
        "torch_version": EXPECTED_RUNTIME["torch_version"],
        "num_fewshot": 0,
        "batch_size": 16,
        "context_length": 512,
        "training_source_commit": EXPECTED_SOURCE_COMMIT,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"EXP-020 official metadata mismatch for {name}: expected {value!r}, got {metadata.get(name)!r}.")
    try:
        metrics = record["raw_lm_eval_result"]["results"][task]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"EXP-020 {task} result lacks raw lm-eval metrics.") from exc
    for name in EXPECTED_LM_METRICS[task]:
        value = _finite(metrics.get(name), name)
        if name.startswith("acc") and "stderr" not in name and not 0.0 <= value <= 1.0:
            raise RuntimeError(f"EXP-020 accuracy {name} is outside [0, 1].")
        if "stderr" in name and value < 0.0:
            raise RuntimeError(f"EXP-020 stderr {name} is negative.")


def validate_wikitext103_record(record: dict[str, Any]) -> None:
    validate_lm_task_metadata(record, "wikitext103")
    result = record.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("EXP-020 WikiText-103 result lacks result data.")
    if _finite(result.get("perplexity"), "perplexity") <= 0.0:
        raise RuntimeError("EXP-020 WikiText-103 perplexity must be positive.")
    if _finite(result.get("bits_per_byte"), "bits_per_byte") <= 0.0:
        raise RuntimeError("EXP-020 WikiText-103 BPB must be positive.")
    if _finite(result.get("mean_negative_log_likelihood"), "mean NLL") <= 0.0:
        raise RuntimeError("EXP-020 WikiText-103 mean NLL must be positive.")
    if not isinstance(result.get("scored_tokens"), int) or result["scored_tokens"] <= 0:
        raise RuntimeError("EXP-020 WikiText-103 requires positive scored_tokens.")


def validate_lm_task_metadata(record: dict[str, Any], task: str) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("EXP-020 official result lacks metadata.")
    expected = {
        "task": task,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "trainable_parameters": EXPECTED_PARAMS,
        "selected_step": EXPECTED_SELECTED_STEP,
        "prediction_tokens": EXPECTED_PREDICTION_TOKENS,
        "device": "cpu",
        "precision": "fp32",
        "cuda_available": False,
        "cuda_visible_devices": "",
        "lm_eval_version": EXPECTED_RUNTIME["lm_eval_version"],
        "datasets_version": EXPECTED_RUNTIME["datasets_version"],
        "torch_version": EXPECTED_RUNTIME["torch_version"],
        "num_fewshot": 0,
        "batch_size": 16,
        "context_length": 512,
        "training_source_commit": EXPECTED_SOURCE_COMMIT,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"EXP-020 official metadata mismatch for {name}: expected {value!r}, got {metadata.get(name)!r}.")
