template_program = '''
import numpy as np


def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Design a heuristic that updates the TSP-GLS edge distance matrix.

    Args:
        edge_distance: Original symmetric distance matrix.
        local_opt_tour: Current local-optimum tour.
        edge_n_used: Historical edge penalty/use matrix.

    Returns:
        A symmetric updated edge distance matrix.
    """
    updated_edge_distance = np.copy(edge_distance)
    penalty_strength = 0.1
    n = len(local_opt_tour)
    for i in range(n):
        node1 = int(local_opt_tour[i])
        node2 = int(local_opt_tour[(i + 1) % n])
        usage = max(0.0, float(edge_n_used[node1, node2]))
        penalty = 1.0 + penalty_strength * (1.0 + usage)
        updated_edge_distance[node1, node2] *= penalty
        updated_edge_distance[node2, node1] = updated_edge_distance[node1, node2]
    return updated_edge_distance
'''

task_description = (
    "Design a heuristic for the TSP guided local search solver. "
    "The heuristic must update the edge distance matrix from the current local-optimum tour "
    "and edge use/penalty matrix. The training objective is to minimize the average optimality gap "
    "on the fixed TSPAEL64 training set."
)
