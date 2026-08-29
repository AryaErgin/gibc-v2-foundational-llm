"""Immutable-stream attribution and fixed-example schedules for EXP-015.

No function here retokenizes, repacks, or changes a frozen token stream.  A
source label is attached to each existing stored token by replaying the
historical EXP-004 mixer, then schedules contain only original window indices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .utils import atomic_json_write, sha256_file

SOURCE_CODES = {"fineweb": 0, "fineweb_edu": 1}
EXPECTED_STREAM_SHA256 = "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1"
EXPECTED_PREDICTION_CONTRIBUTIONS = {"fineweb": 200_017_577, "fineweb_edu": 100_006_231}


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def replay_source_attribution(
    stream_path: Path,
    stored_token_count: int,
    replay: Iterable[tuple[int, str]],
    sidecar_path: Path,
) -> dict[str, object]:
    """Write one source byte per existing stored token and reject any drift."""
    stream = np.memmap(stream_path, mode="r", dtype=np.uint16, shape=(stored_token_count,))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.memmap(sidecar_path, mode="w+", dtype=np.uint8, shape=(stored_token_count,))
    reproduced = hashlib.sha256()
    counts = {source: 0 for source in SOURCE_CODES}
    emitted = 0
    try:
        for position, (token_id, source) in enumerate(replay):
            if position >= stored_token_count:
                raise RuntimeError("Attribution replay exceeds the frozen stored-token count.")
            if source not in SOURCE_CODES:
                raise RuntimeError(f"Unknown attribution source: {source}")
            if int(stream[position]) != int(token_id):
                raise RuntimeError(f"Attribution replay diverges from the frozen stream at stored token {position}.")
            labels[position] = SOURCE_CODES[source]
            reproduced.update(np.asarray([token_id], dtype=np.uint16).tobytes())
            if position:
                counts[source] += 1
            emitted = position + 1
        if emitted != stored_token_count:
            raise RuntimeError(f"Attribution replay ended at {emitted} stored tokens; expected {stored_token_count}.")
        labels.flush()
    finally:
        del labels
        del stream
    expected_hash = sha256_file(stream_path)
    reproduced_hash = reproduced.hexdigest()
    if reproduced_hash != expected_hash:
        raise RuntimeError("Attribution replay token SHA-256 differs from the frozen stream.")
    return {
        "stream_match": True,
        "reproduced_stream_sha256": reproduced_hash,
        "expected_stream_sha256": expected_hash,
        "stored_token_count": stored_token_count,
        "prediction_tokens": stored_token_count - 1,
        "prediction_token_contributions": counts,
        "sidecar_sha256": sha256_file(sidecar_path),
    }


@dataclass(frozen=True)
class FixedExampleSchedule:
    """Three schedules that are permutations of immutable fixed-window IDs."""

    edu_counts: np.ndarray
    static: np.ndarray
    cooldown_edu: np.ndarray
    precooldown_edu: np.ndarray
    low: np.ndarray
    high: np.ndarray
    phase1_windows: int
    block_windows: int
    context_length: int

    @property
    def treatment_contrast(self) -> float:
        return float(self.edu_counts[self.high].mean() - self.edu_counts[self.low].mean()) / self.context_length

    def schedule_hashes(self) -> dict[str, str]:
        return {
            "static": _array_sha256(self.static),
            "cooldown_edu": _array_sha256(self.cooldown_edu),
            "precooldown_edu": _array_sha256(self.precooldown_edu),
            "low": _array_sha256(self.low),
            "high": _array_sha256(self.high),
        }

    def assert_integrity(self) -> None:
        total = self.phase1_windows + 2 * self.block_windows
        expected = np.arange(total, dtype=np.uint32)
        for name, schedule in (("static", self.static), ("cooldown_edu", self.cooldown_edu), ("precooldown_edu", self.precooldown_edu)):
            if schedule.dtype != np.uint32 or schedule.shape != (total,) or not np.array_equal(np.sort(schedule), expected):
                raise RuntimeError(f"{name} is not an exact permutation of original fixed-window IDs.")
        phase1 = expected[: self.phase1_windows]
        if not np.array_equal(self.static[: self.phase1_windows], phase1):
            raise RuntimeError("Static schedule does not retain original phase-one order.")
        if not np.array_equal(self.cooldown_edu[: self.phase1_windows], phase1) or not np.array_equal(self.precooldown_edu[: self.phase1_windows], phase1):
            raise RuntimeError("Treatment schedules do not retain byte-identical original phase one.")
        if not np.array_equal(self.cooldown_edu[self.phase1_windows : self.phase1_windows + self.block_windows], self.low):
            raise RuntimeError("Cooldown-Edu phase two is not LOW.")
        if not np.array_equal(self.cooldown_edu[-self.block_windows :], self.high):
            raise RuntimeError("Cooldown-Edu phase three is not HIGH.")
        if not np.array_equal(self.precooldown_edu[self.phase1_windows : self.phase1_windows + self.block_windows], self.high):
            raise RuntimeError("PreCooldown-Edu phase two is not HIGH.")
        if not np.array_equal(self.precooldown_edu[-self.block_windows :], self.low):
            raise RuntimeError("PreCooldown-Edu phase three is not LOW.")

    @classmethod
    def build(
        cls,
        stored_token_sources: np.ndarray,
        *,
        context_length: int = 512,
        phase1_windows: int = 468_736,
        block_windows: int = 58_624,
        minimum_treatment_contrast: float = 0.15,
    ) -> "FixedExampleSchedule":
        labels = np.asarray(stored_token_sources, dtype=np.uint8)
        total = phase1_windows + 2 * block_windows
        if labels.ndim != 1 or labels.size != total * context_length + 1:
            raise ValueError("Source sidecar length is incompatible with the fixed window horizon.")
        edu_counts = (labels[1:] == SOURCE_CODES["fineweb_edu"]).reshape(total, context_length).sum(axis=1, dtype=np.uint16)
        tail = np.arange(phase1_windows, total, dtype=np.uint32)
        ranked = tail[np.lexsort((tail, edu_counts[tail]))]
        low = np.sort(ranked[:block_windows])
        high = np.sort(ranked[block_windows:])
        schedule = cls(
            edu_counts=edu_counts,
            static=np.arange(total, dtype=np.uint32),
            cooldown_edu=np.concatenate((np.arange(phase1_windows, dtype=np.uint32), low, high)),
            precooldown_edu=np.concatenate((np.arange(phase1_windows, dtype=np.uint32), high, low)),
            low=low,
            high=high,
            phase1_windows=phase1_windows,
            block_windows=block_windows,
            context_length=context_length,
        )
        schedule.assert_integrity()
        if schedule.treatment_contrast < minimum_treatment_contrast:
            raise RuntimeError(f"EXP-015 treatment separation is {schedule.treatment_contrast:.9f}, below {minimum_treatment_contrast:.9f}.")
        return schedule


def schedule_cursor_state(schedule: FixedExampleSchedule, cursor: int, arm: str) -> dict[str, object]:
    """Return checkpointable schedule provenance without changing RunState semantics."""
    schedules = {"A": schedule.static, "B": schedule.cooldown_edu, "C": schedule.precooldown_edu}
    if arm not in schedules or not 0 <= cursor <= len(schedule.static):
        raise ValueError("Invalid EXP-015 arm or schedule cursor.")
    return {"mechanism": "fixed_example_index_permutation", "arm": arm, "schedule_sha256": _array_sha256(schedules[arm]), "next_schedule_cursor": cursor}


def adamw_retention_diagnostic() -> dict[str, object]:
    """Compute the fixed-WSD AdamW final-weight retention diagnostic.

    c_i is descriptive only: it is not a fitted TREC or an EXP-015 design
    input.  Step i uses the learning rate applied by the existing WSD
    scheduler at optimizer update i, and the product is over later updates.
    """
    from .train import WarmupStableDecay

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.0, weight_decay=0.1)
    schedule = WarmupStableDecay(optimizer, 6.0e-4, 6.0e-5, 100, 9_156, 916)
    lrs = np.asarray([schedule.lr_at_step(step) for step in range(1, 9_157)], dtype=np.float64)
    coefficients = np.empty_like(lrs)
    later_retention = 1.0
    for offset in range(len(lrs) - 1, -1, -1):
        coefficients[offset] = lrs[offset] * 0.1 * later_retention
        later_retention *= 1.0 - lrs[offset] * 0.1
    normalized = coefficients / coefficients.max()
    max_step = int(normalized.argmax()) + 1
    values = {str(step): float(normalized[step - 1]) for step in (7_325, 8_240, 8_241, max_step, 8_700, 9_000, 9_156)}
    return {
        "weight_decay": 0.1,
        "max_step": max_step,
        "max_fraction": max_step / 9_156,
        "phase_means": {
            "phase1": float(normalized[:7_324].mean()),
            "phase2": float(normalized[7_324:8_240].mean()),
            "phase3": float(normalized[8_240:].mean()),
        },
        "boundary_values": values,
    }


def indexed_window_sha256(stream_path: Path, schedule_ids: np.ndarray, context_length: int = 512) -> str:
    """Hash (original ID, immutable 513-token row) pairs in ID order."""
    stream = np.memmap(stream_path, mode="r", dtype=np.uint16)
    digest = hashlib.sha256()
    try:
        for index in np.sort(np.asarray(schedule_ids, dtype=np.uint32)):
            start = int(index) * context_length
            digest.update(np.asarray([index], dtype=np.uint32).tobytes())
            digest.update(stream[start : start + context_length + 1].tobytes())
    finally:
        del stream
    return digest.hexdigest()


def analyze_fixed_example_sidecar(sidecar_path: Path, stream_path: Path, output_dir: Path) -> dict[str, object]:
    """Build immutable schedules and complete their non-training audit report."""
    stored_tokens = stream_path.stat().st_size // np.dtype(np.uint16).itemsize
    labels = np.memmap(sidecar_path, mode="r", dtype=np.uint8, shape=(stored_tokens,))
    schedule = FixedExampleSchedule.build(labels)
    del labels
    output_dir.mkdir(parents=True, exist_ok=True)
    named = {"a-static": schedule.static, "b-cooldown-edu": schedule.cooldown_edu, "c-precooldown-edu": schedule.precooldown_edu}
    for name, values in named.items():
        np.save(output_dir / f"schedule-{name}.npy", values, allow_pickle=False)
    tail_counts = schedule.edu_counts[schedule.phase1_windows :]
    def distribution(values: np.ndarray) -> dict[str, object]:
        return {
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "zero": int((values == 0).sum()),
            "full": int((values == schedule.context_length).sum()),
            "mixed": int(((values > 0) & (values < schedule.context_length)).sum()),
        }
    content_hashes = {name: indexed_window_sha256(stream_path, values, schedule.context_length) for name, values in named.items()}
    report = {
        "tail_window_count": int(tail_counts.size),
        "low_count": int(schedule.low.size),
        "high_count": int(schedule.high.size),
        "low_edu_token_share": float(schedule.edu_counts[schedule.low].mean()) / schedule.context_length,
        "high_edu_token_share": float(schedule.edu_counts[schedule.high].mean()) / schedule.context_length,
        "absolute_treatment_contrast": schedule.treatment_contrast,
        "tail_distribution": distribution(tail_counts),
        "low_distribution": distribution(schedule.edu_counts[schedule.low]),
        "high_distribution": distribution(schedule.edu_counts[schedule.high]),
        "schedule_hashes": schedule.schedule_hashes(),
        "indexed_window_sha256": content_hashes,
        "static_original_equivalence": bool(np.array_equal(schedule.static, np.arange(len(schedule.static), dtype=np.uint32))),
        "phase1_equivalence": bool(np.array_equal(schedule.cooldown_edu[: schedule.phase1_windows], schedule.precooldown_edu[: schedule.phase1_windows])),
        "block_swap_equivalence": bool(np.array_equal(schedule.cooldown_edu[schedule.phase1_windows :], np.concatenate((schedule.low, schedule.high))) and np.array_equal(schedule.precooldown_edu[schedule.phase1_windows :], np.concatenate((schedule.high, schedule.low)))),
        "all_example_membership_equality": all(np.array_equal(np.sort(values), schedule.static) for values in named.values()),
        "adamw_retention": adamw_retention_diagnostic(),
    }
    return report


def finalize_preflight_report(artifact_dir: Path, output_dir: Path) -> dict[str, object]:
    """Merge the completed immutable-window audit into the replay report."""
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    report_path = output_dir / "preflight.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    report.update(analyze_fixed_example_sidecar(output_dir / "stored-token-sources.uint8", artifact_dir / manifest["packed"]["train_stream_file"], output_dir))
    report["validation_hashes"] = {
        "general_inputs": manifest["general_validation"]["inputs_sha256"],
        "general_targets": manifest["general_validation"]["targets_sha256"],
        "edu_inputs": manifest["edu_validation"]["inputs_sha256"],
        "edu_targets": manifest["edu_validation"]["targets_sha256"],
    }
    atomic_json_write(report_path, report)
    return report


def reconstruct_exp004_attribution(config: Any, artifact_dir: Path, output_dir: Path) -> dict[str, object]:
    """Replay EXP-004 against its immutable stream and materialize only sidecars.

    The original stream is opened read-only.  Replaying requires the exact
    source revisions and cached contamination index used by EXP-004; any token
    mismatch raises before schedules are written.
    """
    from .data import NgramContaminationFilter
    from .exp004 import GlobalDeduplicatedTokenMixer, SOURCE_ORDER, _screened_train_documents
    from .tokenizer import load_tokenizer

    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    packed = manifest["packed"]
    stream_path = artifact_dir / packed["train_stream_file"]
    if packed["train_stream_sha256"] != EXPECTED_STREAM_SHA256 or sha256_file(stream_path) != EXPECTED_STREAM_SHA256:
        raise RuntimeError("EXP-015 requires the exact immutable EXP-004 stream.")
    if manifest["mixture"].get("actual_prediction_token_contributions") != EXPECTED_PREDICTION_CONTRIBUTIONS:
        raise RuntimeError("EXP-015 requires the recorded original source-token totals.")
    index_path = artifact_dir / "cache" / "benchmarks" / "benchmark-ngrams.sqlite"
    if not index_path.is_file():
        raise RuntimeError("EXP-015 reconstruction requires the original cached benchmark n-gram index.")
    tokenizer_path = artifact_dir / manifest["tokenizer"]["path"]
    tokenizer = load_tokenizer(tokenizer_path)
    eod_id = tokenizer.token_to_id(config.data.eod_token)
    if eod_id is None:
        raise RuntimeError("EXP-015 reconstruction cannot find the frozen EOD token.")
    contamination_filter = NgramContaminationFilter(None, config.data.contamination_ngram_size, sqlite_path=index_path)
    source_configs = {
        "fineweb": config.data,
        "fineweb_edu": replace(config.data, dataset_repo="HuggingFaceFW/fineweb-edu", dataset_config="default", dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    }
    counters = {source: {"scanned_documents": 0, "accepted_documents": 0, "rejected_documents": 0, "validation_documents_excluded": 0} for source in SOURCE_ORDER}
    sources = {source: _screened_train_documents(source_configs[source], artifact_dir / "cache" / source, contamination_filter, counters[source]) for source in SOURCE_ORDER}
    mixer = GlobalDeduplicatedTokenMixer(sources, tokenizer, eod_id, dict(config.mixture["target_prediction_tokens"]), int(packed["train_token_count_including_final_target"]))
    sidecar_path = output_dir / "stored-token-sources.uint8"
    replay = replay_source_attribution(stream_path, int(packed["train_token_count_including_final_target"]), mixer.iter_with_sources(), sidecar_path)
    if replay["prediction_token_contributions"] != EXPECTED_PREDICTION_CONTRIBUTIONS:
        raise RuntimeError("EXP-015 replay source totals differ from the original immutable stream.")
    labels = np.memmap(sidecar_path, mode="r", dtype=np.uint8, shape=(int(packed["train_token_count_including_final_target"]),))
    schedule = FixedExampleSchedule.build(labels)
    del labels
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "schedule-a-static.npy", schedule.static, allow_pickle=False)
    np.save(output_dir / "schedule-b-cooldown-edu.npy", schedule.cooldown_edu, allow_pickle=False)
    np.save(output_dir / "schedule-c-precooldown-edu.npy", schedule.precooldown_edu, allow_pickle=False)
    report = {
        "experiment_id": "EXP-015",
        "replay": replay,
        "source_counters": counters,
        "tail_window_count": 2 * schedule.block_windows,
        "low_count": int(len(schedule.low)),
        "high_count": int(len(schedule.high)),
        "low_edu_token_share": float(schedule.edu_counts[schedule.low].mean()) / schedule.context_length,
        "high_edu_token_share": float(schedule.edu_counts[schedule.high].mean()) / schedule.context_length,
        "absolute_treatment_contrast": schedule.treatment_contrast,
        "schedule_hashes": schedule.schedule_hashes(),
        "phase1_windows": schedule.phase1_windows,
        "block_windows": schedule.block_windows,
        "static_original_equivalence": bool(np.array_equal(schedule.static, np.arange(len(schedule.static), dtype=np.uint32))),
    }
    atomic_json_write(output_dir / "preflight.json", report)
    return report
