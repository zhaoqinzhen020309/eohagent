from __future__ import annotations

import types
from typing import Any

import numpy as np


ARCHITECTURE_CLASS_NAME = "CandidateGLSArchitecture"
ARCHITECTURE_LAYERS = {
    "distance_update",
    "penalty_edge_selection",
    "perturbation_schedule",
    "restart_policy",
}


def load_architecture(code: str):
    module = types.ModuleType("candidate_gls_architecture")
    exec(code, module.__dict__)
    architecture_cls = getattr(module, ARCHITECTURE_CLASS_NAME)
    return architecture_cls()


def normalize_penalty_edges(edges: Any, n_nodes: int) -> list[tuple[int, int]]:
    arr = np.asarray(edges, dtype=int)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("select_penalty_edges must return an array-like object with shape (k, 2)")
    if len(arr) == 0:
        raise ValueError("select_penalty_edges must return at least one edge")
    max_edges = min(20, max(1, n_nodes * (n_nodes - 1) // 2))
    if len(arr) > max_edges:
        raise ValueError("select_penalty_edges returned too many edges")

    seen = set()
    normalized = []
    for raw_a, raw_b in arr.tolist():
        a = int(raw_a)
        b = int(raw_b)
        if a == b:
            raise ValueError("select_penalty_edges returned a self-loop")
        if a < 0 or b < 0 or a >= n_nodes or b >= n_nodes:
            raise ValueError("select_penalty_edges returned an out-of-range node index")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError("select_penalty_edges returned duplicate undirected edges")
        seen.add(key)
        normalized.append((a, b))
    return normalized


def validate_architecture_code(code: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        architecture = load_architecture(code)
    except Exception as exc:
        return [f"architecture_load_error: {exc}"], warnings

    for method_name in [
        "update_edge_distance",
        "select_penalty_edges",
        "perturbation_schedule",
        "restart_policy",
    ]:
        method = getattr(architecture, method_name, None)
        if not callable(method):
            errors.append(f"architecture_contract_error: missing callable method {method_name}")
    if errors:
        return errors, warnings

    edge_distance = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ],
        dtype=float,
    )
    original_edge_distance = edge_distance.copy()
    local_opt_tour = np.array([0, 1, 2, 3], dtype=int)
    best_tour = np.array([0, 2, 1, 3], dtype=int)
    edge_n_used = np.zeros_like(edge_distance)
    edge_n_used[0, 1] = edge_n_used[1, 0] = 2.0
    edge_n_used[1, 2] = edge_n_used[2, 1] = 3.0

    try:
        guided = np.asarray(
            architecture.update_edge_distance(edge_distance, local_opt_tour, edge_n_used),
            dtype=float,
        )
        if guided.shape != edge_distance.shape:
            errors.append(f"output_shape_error: expected shape {edge_distance.shape}, got {guided.shape}")
        if not np.all(np.isfinite(guided)):
            errors.append("nan_inf_error: update_edge_distance output contains non-finite values")
        if np.any(guided < 0):
            errors.append("negative_distance_error: updated edge distances contain negative values")
        if not np.allclose(guided, guided.T, rtol=1e-8, atol=1e-8):
            errors.append("symmetry_error: updated edge distance matrix is not symmetric")
        if not np.allclose(edge_distance, original_edge_distance, rtol=1e-12, atol=1e-12):
            errors.append("in_place_error: candidate modified edge_distance input in place")

        edge_gap = guided - edge_distance
        penalty_edges = architecture.select_penalty_edges(
            edge_distance,
            guided,
            edge_gap,
            local_opt_tour,
            best_tour,
            edge_n_used,
            1,
        )
        normalize_penalty_edges(penalty_edges, edge_distance.shape[0])

        moves = int(architecture.perturbation_schedule(1, edge_distance.shape[0], 1))
        if moves < 1 or moves > 20:
            errors.append("perturbation_schedule_error: returned moves must be in [1, 20]")

        restart = architecture.restart_policy(50, 100.0, 90.0, 10)
        if not isinstance(restart, (bool, np.bool_)):
            errors.append("restart_policy_error: restart_policy must return a boolean")
    except Exception as exc:
        errors.append(f"runtime_error: architecture dummy call failed: {exc}")

    return errors, warnings
