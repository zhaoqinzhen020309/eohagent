from __future__ import annotations

import math

import numpy as np

from eohagent.problems.tsp_gls_arch_spec.spec_schema import canonical_arch_spec


class SpecCandidateGLSArchitecture:
    def __init__(self, arch_spec):
        self.arch_spec = canonical_arch_spec(arch_spec)
        self.iteration = 0
        self.n_nodes = 0
        self.stagnation_count = 0

    def set_search_state(self, iteration: int, n_nodes: int, stagnation_count: int) -> None:
        self.iteration = int(iteration)
        self.n_nodes = int(n_nodes)
        self.stagnation_count = int(stagnation_count)

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        edge_distance = np.asarray(edge_distance, dtype=float)
        updated = edge_distance.copy()
        policy = self.arch_spec["distance_update_policy"]
        tour_edges = self._tour_edges(local_opt_tour)
        mean_length = self._mean_tour_length(edge_distance, tour_edges)
        usages = np.asarray([max(0.0, float(edge_n_used[a, b])) for a, b in tour_edges], dtype=float)
        lengths = np.asarray([max(1e-12, float(edge_distance[a, b])) for a, b in tour_edges], dtype=float)
        threshold = self._rank_threshold(usages, self.arch_spec["penalty_policy"]["top_k"])
        mean_utility = float(np.mean(lengths / (1.0 + usages))) + 1e-12
        multiplier = self._schedule_multiplier()

        for a, b in tour_edges:
            usage = max(0.0, float(edge_n_used[a, b]))
            length = max(1e-12, float(edge_distance[a, b]))
            signal = self._distance_signal(policy["type"], usage, length, mean_length, mean_utility, threshold)
            signal = max(0.0, signal) ** float(policy["usage_power"])
            if policy["length_weight"] > 0.0:
                signal *= max(0.2, min(5.0, length / mean_length)) ** policy["length_weight"]
            penalty = 1.0 + policy["strength"] * multiplier * signal
            penalty = min(policy["cap_multiplier"], max(1.0, penalty))
            updated[a, b] = edge_distance[a, b] * penalty
            updated[b, a] = edge_distance[b, a] * penalty
        return updated

    def select_penalty_edges(
        self,
        edge_weight,
        edge_weight_guided,
        edge_weight_gap,
        local_opt_tour,
        best_tour,
        edge_penalty,
        iteration,
    ):
        policy = self.arch_spec["penalty_policy"]
        tour_edges = self._tour_edges(local_opt_tour)
        scores = []
        gap_matrix = np.asarray(edge_weight_gap, dtype=float)
        edge_weight = np.asarray(edge_weight, dtype=float)
        for a, b in tour_edges:
            gap = max(0.0, float(gap_matrix[a, b]))
            usage = max(0.0, float(edge_penalty[a, b]))
            utility = float(edge_weight[a, b]) / (1.0 + usage)
            if policy["edge_selection"] == "top_gap":
                score = gap
            elif policy["edge_selection"] == "tour_utility":
                score = utility
            elif policy["edge_selection"] == "rank_topk":
                score = usage + 1e-9 * utility
            elif policy["edge_selection"] == "mixed_gap_utility":
                score = gap / (float(np.mean(gap_matrix[gap_matrix > 0])) + 1e-12) + utility / (
                    float(np.mean(edge_weight[edge_weight > 0])) + 1e-12
                )
            else:
                raise ValueError(f"unsupported edge_selection: {policy['edge_selection']}")
            scores.append((score, a, b))
        scores.sort(reverse=True)
        limit = min(policy["top_k"], len(scores))
        return np.asarray([(a, b) for _, a, b in scores[:limit]], dtype=int)

    def penalty_increment(self, a: int, b: int, iteration: int, edge_penalty) -> float:
        return float(self.arch_spec["penalty_policy"]["increment"])

    def penalty_decay(self) -> float:
        return float(self.arch_spec["penalty_policy"]["decay"])

    def perturbation_schedule(self, iteration, n_nodes, base_perturbation_moves):
        policy = self.arch_spec["escape_policy"]
        moves = int(policy["moves"])
        if policy["trigger"] == "periodic" and int(iteration) > 0 and int(iteration) % int(policy["period"]) == 0:
            moves += 1
        elif policy["trigger"] == "stagnation" and self.stagnation_count >= int(policy["stagnation_threshold"]):
            moves += 1
        elif policy["trigger"] == "size_adaptive" and int(n_nodes) >= int(policy["size_threshold"]):
            moves += 1
        return int(min(policy["max_moves"], max(1, moves)))

    def restart_policy(self, iteration, current_cost, best_cost, stagnation_count):
        return self.restart_action(iteration, current_cost, best_cost, stagnation_count) != "none"

    def restart_action(self, iteration, current_cost, best_cost, stagnation_count) -> str:
        policy = self.arch_spec["restart_policy"]
        if policy["action"] == "none":
            return "none"
        triggered = False
        if policy["trigger"] == "periodic":
            triggered = int(iteration) > 0 and int(iteration) % int(policy["period"]) == 0
        elif policy["trigger"] == "stagnation":
            triggered = int(stagnation_count) >= int(policy["stagnation_threshold"])
        if not triggered:
            return "none"
        return str(policy["action"])

    def restart_penalty_decay(self) -> float:
        return float(self.arch_spec["restart_policy"]["penalty_decay"])

    def _schedule_multiplier(self) -> float:
        policy = self.arch_spec["schedule_policy"]
        schedule = policy["strength_schedule"]
        if schedule == "constant":
            return 1.0
        if schedule == "decay_after_early":
            return 1.0 if self.iteration <= policy["early_iterations"] else float(policy["late_multiplier"])
        if schedule == "stagnation_boost":
            return 1.0 if self.stagnation_count == 0 else float(policy["stagnation_multiplier"])
        if schedule == "size_scaled":
            return float(policy["size_multiplier"]) if self.n_nodes >= int(policy["size_threshold"]) else 1.0
        raise ValueError(f"unsupported strength_schedule: {schedule}")

    @staticmethod
    def _tour_edges(local_opt_tour) -> list[tuple[int, int]]:
        tour = [int(node) for node in list(local_opt_tour)]
        if len(tour) < 2:
            raise ValueError("local_opt_tour must contain at least two nodes")
        return [(tour[index], tour[(index + 1) % len(tour)]) for index in range(len(tour))]

    @staticmethod
    def _mean_tour_length(edge_distance, tour_edges) -> float:
        lengths = [max(1e-12, float(edge_distance[a, b])) for a, b in tour_edges]
        return float(np.mean(lengths)) + 1e-12

    @staticmethod
    def _rank_threshold(values, top_k: int) -> float:
        if len(values) == 0:
            return 0.0
        k = min(max(1, int(top_k)), len(values))
        return float(np.partition(values, -k)[-k])

    @staticmethod
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
        raise ValueError(f"unsupported distance_update_policy.type: {distance_type}")
