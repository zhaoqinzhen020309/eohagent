from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from eohagent.core.gls_architecture import normalize_penalty_edges
from eohagent.problems.tsp_gls_arch import gls_operators, gls_utils
from eohagent.problems.tsp_gls_arch_spec.gls_arch_spec_evol import local_search, nearest_neighbor_2End, route2tour
from eohagent.problems.tsp_gls_solver_tree.tree_schema import canonical_solver_tree


@dataclass
class TreeSearchState:
    edge_weight: Any
    nearest_indices: Any
    initial_route: Any
    initial_cost: float
    cur_route: Any
    cur_cost: float
    best_route: Any
    best_cost: float
    edge_penalty: Any
    iteration: int
    stagnation_count: int
    first_improvement: bool


def guided_local_search(
    coords,
    edge_weight,
    nearest_indices,
    init_tour,
    init_cost,
    t_lim,
    ite_max,
    perturbation_moves,
    first_improvement=False,
    solver_tree=None,
):
    del coords, perturbation_moves
    random.seed(2024)
    tree = canonical_solver_tree(solver_tree)
    cur_route, cur_cost = init_tour, init_cost
    length = len(edge_weight[0])
    state = TreeSearchState(
        edge_weight=edge_weight,
        nearest_indices=nearest_indices,
        initial_route=cur_route.copy(),
        initial_cost=float(cur_cost),
        cur_route=cur_route,
        cur_cost=float(cur_cost),
        best_route=cur_route.copy(),
        best_cost=float(cur_cost),
        edge_penalty=np.zeros((length, length)),
        iteration=0,
        stagnation_count=0,
        first_improvement=bool(first_improvement),
    )

    while state.iteration < int(ite_max) and time.time() < t_lim:
        _execute_node(tree["body"], state)
        state.cur_cost = gls_utils.tour_cost_2End(edge_weight, state.cur_route)
        if state.cur_cost < state.best_cost:
            state.best_route, state.best_cost = state.cur_route, state.cur_cost
            state.stagnation_count = 0
        else:
            state.stagnation_count += 1
        state.iteration += 1
    return state.best_route, state.best_cost, state.iteration


def _execute_node(node: dict[str, Any], state: TreeSearchState) -> None:
    node_type = node["type"]
    if node_type == "sequence":
        for child in node["steps"]:
            _execute_node(child, state)
        return
    if node_type == "if_stagnated":
        if state.stagnation_count >= int(node["threshold"]):
            _execute_node(node["then"], state)
        return
    if node_type == "local_search":
        state.cur_route, state.cur_cost = local_search(
            state.cur_route,
            state.cur_cost,
            state.edge_weight,
            state.nearest_indices,
            bool(node["first_improvement"]),
        )
        return
    if node_type == "penalty_step":
        _execute_penalty_step(node, state)
        return
    if node_type == "restart":
        _execute_restart(node, state)
        return
    raise ValueError(f"unsupported solver tree execution node: {node_type}")


def _execute_penalty_step(node: dict[str, Any], state: TreeSearchState) -> None:
    penalty = node["penalty"]
    decay = float(penalty["decay"])
    if decay > 0.0:
        state.edge_penalty *= max(0.0, min(1.0, 1.0 - decay))
    repeat = int(node["repeat"])
    length = len(state.edge_weight[0])
    for _ in range(repeat):
        cur_tour = route2tour(state.cur_route)
        best_tour = route2tour(state.best_route)
        guided = _updated_edge_distance(
            state.edge_weight,
            np.array(cur_tour),
            state.edge_penalty,
            node,
            state.iteration,
            state.stagnation_count,
            length,
        )
        guided = np.asmatrix(guided)
        gap = guided - state.edge_weight
        penalty_edges = _select_penalty_edges(
            state.edge_weight,
            guided,
            gap,
            np.array(cur_tour),
            np.array(best_tour),
            state.edge_penalty,
            node,
        )
        for rows, columns in normalize_penalty_edges(penalty_edges, length):
            increment = float(penalty["increment"])
            if not np.isfinite(increment) or increment <= 0:
                raise ValueError("penalty increment must be positive and finite")
            state.edge_penalty[rows, columns] += increment
            state.edge_penalty[columns, rows] += increment
            for node_id in [rows, columns]:
                delta, new_route = gls_operators.two_opt_o2a_all(
                    state.cur_route,
                    guided,
                    state.nearest_indices,
                    node_id,
                )
                if delta < 0:
                    state.cur_cost = gls_utils.tour_cost_2End(state.edge_weight, new_route)
                    state.cur_route = new_route
                delta, new_route = gls_operators.relocate_o2a_all(
                    state.cur_route,
                    guided,
                    state.nearest_indices,
                    node_id,
                )
                if delta < 0:
                    state.cur_cost = gls_utils.tour_cost_2End(state.edge_weight, new_route)
                    state.cur_route = new_route


def _execute_restart(node: dict[str, Any], state: TreeSearchState) -> None:
    action = node["action"]
    if action == "reset_to_best":
        state.cur_route, state.cur_cost = state.best_route, state.best_cost
    elif action == "reset_to_initial":
        state.cur_route, state.cur_cost = state.initial_route.copy(), state.initial_cost
    elif action == "decay_penalty":
        decay = float(node["penalty_decay"])
        state.edge_penalty *= max(0.0, min(1.0, 1.0 - decay))
    elif action == "rebuild_nearest":
        depot = int(state.iteration % len(state.edge_weight[0]))
        state.cur_route = nearest_neighbor_2End(state.edge_weight, depot).astype(int)
        state.cur_cost = gls_utils.tour_cost_2End(state.edge_weight, state.cur_route)
    else:
        raise ValueError(f"unsupported restart action: {action}")


def _updated_edge_distance(edge_distance, local_opt_tour, edge_penalty, node, iteration: int, stagnation_count: int, n_nodes: int):
    edge_distance = np.asarray(edge_distance, dtype=float)
    updated = edge_distance.copy()
    distance = node["distance_update"]
    penalty = node["penalty"]
    schedule = node["schedule"]
    tour_edges = _tour_edges(local_opt_tour)
    mean_length = _mean_tour_length(edge_distance, tour_edges)
    usages = np.asarray([max(0.0, float(edge_penalty[a, b])) for a, b in tour_edges], dtype=float)
    lengths = np.asarray([max(1e-12, float(edge_distance[a, b])) for a, b in tour_edges], dtype=float)
    threshold = _rank_threshold(usages, penalty["top_k"])
    mean_utility = float(np.mean(lengths / (1.0 + usages))) + 1e-12
    multiplier = _schedule_multiplier(schedule, iteration, stagnation_count, n_nodes)
    for a, b in tour_edges:
        usage = max(0.0, float(edge_penalty[a, b]))
        length = max(1e-12, float(edge_distance[a, b]))
        signal = _distance_signal(distance["type"], usage, length, mean_length, mean_utility, threshold)
        signal = max(0.0, signal) ** float(distance["usage_power"])
        if distance["length_weight"] > 0.0:
            signal *= max(0.2, min(5.0, length / mean_length)) ** float(distance["length_weight"])
        edge_multiplier = 1.0 + float(distance["strength"]) * multiplier * signal
        edge_multiplier = min(float(distance["cap_multiplier"]), max(1.0, edge_multiplier))
        updated[a, b] = edge_distance[a, b] * edge_multiplier
        updated[b, a] = edge_distance[b, a] * edge_multiplier
    return updated


def _select_penalty_edges(edge_weight, edge_weight_guided, edge_weight_gap, local_opt_tour, best_tour, edge_penalty, node):
    del edge_weight_guided, best_tour
    penalty = node["penalty"]
    tour_edges = _tour_edges(local_opt_tour)
    scores = []
    gap_matrix = np.asarray(edge_weight_gap, dtype=float)
    edge_weight = np.asarray(edge_weight, dtype=float)
    for a, b in tour_edges:
        gap = max(0.0, float(gap_matrix[a, b]))
        usage = max(0.0, float(edge_penalty[a, b]))
        utility = float(edge_weight[a, b]) / (1.0 + usage)
        if penalty["edge_selection"] == "top_gap":
            score = gap
        elif penalty["edge_selection"] == "tour_utility":
            score = utility
        elif penalty["edge_selection"] == "rank_topk":
            score = usage + 1e-9 * utility
        elif penalty["edge_selection"] == "mixed_gap_utility":
            score = gap / (float(np.mean(gap_matrix[gap_matrix > 0])) + 1e-12) + utility / (
                float(np.mean(edge_weight[edge_weight > 0])) + 1e-12
            )
        else:
            raise ValueError(f"unsupported edge_selection: {penalty['edge_selection']}")
        scores.append((score, a, b))
    scores.sort(reverse=True)
    limit = min(int(penalty["top_k"]), len(scores))
    return np.asarray([(a, b) for _, a, b in scores[:limit]], dtype=int)


def _schedule_multiplier(schedule: dict[str, Any], iteration: int, stagnation_count: int, n_nodes: int) -> float:
    kind = schedule["strength_schedule"]
    if kind == "constant":
        return 1.0
    if kind == "decay_after_early":
        return 1.0 if iteration <= int(schedule["early_iterations"]) else float(schedule["late_multiplier"])
    if kind == "stagnation_boost":
        return 1.0 if stagnation_count == 0 else float(schedule["stagnation_multiplier"])
    if kind == "size_scaled":
        return float(schedule["size_multiplier"]) if n_nodes >= int(schedule["size_threshold"]) else 1.0
    raise ValueError(f"unsupported strength_schedule: {kind}")


def _tour_edges(local_opt_tour) -> list[tuple[int, int]]:
    tour = [int(node) for node in list(local_opt_tour)]
    if len(tour) < 2:
        raise ValueError("local_opt_tour must contain at least two nodes")
    return [(tour[index], tour[(index + 1) % len(tour)]) for index in range(len(tour))]


def _mean_tour_length(edge_distance, tour_edges) -> float:
    lengths = [max(1e-12, float(edge_distance[a, b])) for a, b in tour_edges]
    return float(np.mean(lengths)) + 1e-12


def _rank_threshold(values, top_k: int) -> float:
    if len(values) == 0:
        return 0.0
    k = min(max(1, int(top_k)), len(values))
    return float(np.partition(values, -k)[-k])


def _distance_signal(distance_type: str, usage: float, length: float, mean_length: float, mean_utility: float, threshold: float) -> float:
    if distance_type == "linear_usage":
        return 1.0 + usage
    if distance_type == "sqrt_usage":
        return math.sqrt(1.0 + usage)
    if distance_type == "log_usage":
        return math.log1p(1.0 + usage)
    if distance_type == "length_weighted_usage":
        return (1.0 + usage) * max(0.2, min(5.0, length / mean_length))
    if distance_type == "utility_ratio":
        return max(0.1, min(5.0, (length / (1.0 + usage)) / mean_utility))
    if distance_type == "rank_usage":
        return 2.0 if usage >= threshold else 0.5
    if distance_type == "smooth_saturation":
        return 1.0 + usage / (1.0 + usage)
    raise ValueError(f"unsupported distance_update.type: {distance_type}")

