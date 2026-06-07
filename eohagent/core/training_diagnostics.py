from __future__ import annotations

import math
from typing import Any

import numpy as np


def build_training_diagnosis(evidence: dict[str, Any] | None, parent_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    gaps = _gaps(evidence)
    if gaps.size == 0:
        return {}
    parent_gaps = _gaps(parent_evidence)
    diagnosis = {
        "mean_gap": _round(float(np.mean(gaps))),
        "std_gap": _round(float(np.std(gaps))),
        "tail_gap": _tail_gap(gaps),
        "max_gap": _round(float(np.max(gaps))),
        "worst_training_instances": _indexed_values(gaps, largest=True),
        "best_training_instances": _indexed_values(gaps, largest=False),
        "failure_signature": _failure_signature(gaps),
    }
    if parent_gaps.size == gaps.size:
        delta = gaps - parent_gaps
        diagnosis["improved_instances_vs_parent"] = _delta_instances(delta, improved=True)
        diagnosis["worsened_instances_vs_parent"] = _delta_instances(delta, improved=False)
        diagnosis["mean_delta_vs_parent"] = _round(float(np.mean(delta)))
        diagnosis["tail_delta_vs_parent"] = _round(_tail_gap(gaps) - _tail_gap(parent_gaps))
    return diagnosis


def compact_training_diagnosis(diagnosis: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diagnosis, dict):
        return {}
    return {
        "mean_gap": diagnosis.get("mean_gap"),
        "std_gap": diagnosis.get("std_gap"),
        "tail_gap": diagnosis.get("tail_gap"),
        "max_gap": diagnosis.get("max_gap"),
        "failure_signature": diagnosis.get("failure_signature"),
        "worst_training_instances": (diagnosis.get("worst_training_instances") or [])[:3],
        "improved_instances_vs_parent": (diagnosis.get("improved_instances_vs_parent") or [])[:3],
        "worsened_instances_vs_parent": (diagnosis.get("worsened_instances_vs_parent") or [])[:3],
    }


def _gaps(evidence: dict[str, Any] | None) -> np.ndarray:
    if not isinstance(evidence, dict):
        return np.asarray([], dtype=float)
    values = evidence.get("per_instance_gap") or []
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _tail_gap(gaps: np.ndarray) -> float:
    count = max(1, int(math.ceil(0.2 * gaps.size)))
    tail = np.sort(gaps)[-count:]
    return _round(float(np.mean(tail)))


def _indexed_values(gaps: np.ndarray, largest: bool) -> list[dict[str, Any]]:
    order = np.argsort(gaps)
    if largest:
        order = order[::-1]
    return [{"index": int(i), "gap": _round(float(gaps[i]))} for i in order[:5]]


def _delta_instances(delta: np.ndarray, improved: bool) -> list[dict[str, Any]]:
    order = np.argsort(delta)
    if not improved:
        order = order[::-1]
    items = []
    for index in order:
        value = float(delta[index])
        if improved and value >= -1e-10:
            continue
        if not improved and value <= 1e-10:
            continue
        items.append({"index": int(index), "delta_gap": _round(value)})
        if len(items) >= 5:
            break
    return items


def _failure_signature(gaps: np.ndarray) -> str:
    mean = float(np.mean(gaps))
    std = float(np.std(gaps))
    max_gap = float(np.max(gaps))
    tail = _tail_gap(gaps)
    if max_gap > max(0.25, mean + 3.0 * std):
        return "single_extreme_training_failure"
    if tail > mean + std:
        return "tail_heavy_training_failures"
    if std > max(0.05, mean):
        return "high_variance_training_behavior"
    return "broadly_stable_training_behavior"


def _round(value: float) -> float:
    return round(float(value), 8)
