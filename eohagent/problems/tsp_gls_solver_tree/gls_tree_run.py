import time

import numpy as np

from eohagent.problems.tsp_gls_arch import gls_utils
from eohagent.problems.tsp_gls_solver_tree import gls_tree_evol


def solve_instance(n, opt_cost, dis_matrix, coord, time_limit, ite_max, perturbation_moves, heuristic):
    time.sleep(1)
    t = time.time()

    architecture = heuristic.CandidateGLSArchitecture()
    init_tour = gls_tree_evol.nearest_neighbor_2End(dis_matrix, 0).astype(int)
    init_cost = gls_utils.tour_cost_2End(dis_matrix, init_tour)
    nb = 100
    nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb + 1].astype(int)

    best_tour, best_cost, iter_i = gls_tree_evol.guided_local_search(
        coord,
        dis_matrix,
        nearest_indices,
        init_tour,
        init_cost,
        t + time_limit,
        ite_max,
        perturbation_moves,
        first_improvement=False,
        solver_tree=architecture.solver_tree,
    )

    gap = (best_cost / opt_cost - 1) * 100
    return gap

