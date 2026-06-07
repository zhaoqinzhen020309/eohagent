import random

import numpy as np
from numba import jit

from eohagent.problems.tsp_gls_arch import gls_operators, gls_utils
from eohagent.core.gls_architecture import normalize_penalty_edges


def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    route2End = np.zeros((n, 2))
    route2End[0, 0] = tour[-2]
    route2End[0, 1] = tour[1]
    for i in range(1, n):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    return route2End


@jit(nopython=True)
def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False
        delta, new_tour = gls_operators.two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

        delta, new_tour = gls_operators.relocate_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

    return cur_route, cur_cost


@jit(nopython=True)
def route2tour(route):
    s = 0
    tour = []
    for i in range(len(route)):
        tour.append(route[s, 1])
        s = route[s, 1]
    return tour


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
    guide_algorithm=None,
):
    random.seed(2024)

    cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight, nearest_indices, first_improvement)
    best_route, best_cost = cur_route, cur_cost
    initial_route, initial_cost = cur_route.copy(), cur_cost
    length = len(edge_weight[0])
    iter_i = 0
    stagnation_count = 0
    edge_penalty = np.zeros((length, length))

    while iter_i < ite_max:
        if hasattr(guide_algorithm, "set_search_state"):
            guide_algorithm.set_search_state(iter_i, length, stagnation_count)

        decay = float(guide_algorithm.penalty_decay()) if hasattr(guide_algorithm, "penalty_decay") else 0.0
        if decay > 0.0:
            edge_penalty *= max(0.0, min(1.0, 1.0 - decay))

        moves = int(guide_algorithm.perturbation_schedule(iter_i, length, perturbation_moves))
        if moves < 1 or moves > 20:
            raise ValueError("perturbation_schedule returned moves outside [1, 20]")

        for move in range(moves):
            cur_tour = route2tour(cur_route)
            best_tour = route2tour(best_route)
            edge_weight_guided = guide_algorithm.update_edge_distance(
                edge_weight,
                np.array(cur_tour),
                edge_penalty,
            )
            edge_weight_guided = np.asmatrix(edge_weight_guided)
            edge_weight_gap = edge_weight_guided - edge_weight
            penalty_edges = guide_algorithm.select_penalty_edges(
                edge_weight,
                edge_weight_guided,
                edge_weight_gap,
                np.array(cur_tour),
                np.array(best_tour),
                edge_penalty,
                iter_i,
            )

            for rows, columns in normalize_penalty_edges(penalty_edges, length):
                increment = 1.0
                if hasattr(guide_algorithm, "penalty_increment"):
                    increment = float(guide_algorithm.penalty_increment(rows, columns, iter_i, edge_penalty))
                if not np.isfinite(increment) or increment <= 0:
                    raise ValueError("penalty_increment must be positive and finite")
                edge_penalty[rows, columns] += increment
                edge_penalty[columns, rows] += increment

                for node_id in [rows, columns]:
                    delta, new_route = gls_operators.two_opt_o2a_all(
                        cur_route,
                        edge_weight_guided,
                        nearest_indices,
                        node_id,
                    )
                    if delta < 0:
                        cur_cost = gls_utils.tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route
                    delta, new_route = gls_operators.relocate_o2a_all(
                        cur_route,
                        edge_weight_guided,
                        nearest_indices,
                        node_id,
                    )
                    if delta < 0:
                        cur_cost = gls_utils.tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route

        cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight, nearest_indices, first_improvement)
        cur_cost = gls_utils.tour_cost_2End(edge_weight, cur_route)

        if cur_cost < best_cost:
            best_route, best_cost = cur_route, cur_cost
            stagnation_count = 0
        else:
            stagnation_count += 1
        iter_i += 1

        action = "reset_to_best" if bool(guide_algorithm.restart_policy(iter_i, cur_cost, best_cost, stagnation_count)) else "none"
        if hasattr(guide_algorithm, "restart_action"):
            action = guide_algorithm.restart_action(iter_i, cur_cost, best_cost, stagnation_count)
        if action == "reset_to_best":
            cur_route, cur_cost = best_route, best_cost
        elif action == "reset_to_initial":
            cur_route, cur_cost = initial_route.copy(), initial_cost
        elif action == "decay_penalty":
            decay_amount = 0.5
            if hasattr(guide_algorithm, "restart_penalty_decay"):
                decay_amount = float(guide_algorithm.restart_penalty_decay())
            edge_penalty *= max(0.0, min(1.0, 1.0 - decay_amount))
        elif action == "rebuild_nearest":
            depot = int(iter_i % length)
            cur_route = nearest_neighbor_2End(edge_weight, depot).astype(int)
            cur_cost = gls_utils.tour_cost_2End(edge_weight, cur_route)
        elif action != "none":
            raise ValueError(f"unsupported restart action: {action}")

    return best_route, best_cost, iter_i

