# Second best heuristic extracted from training results
# Objective: 0.00624
# Algorithm: This algorithm calculates scores by combining a bin's remaining capacity with a Gaussian-based penalty for new bins, an exponential bonus for utilization, a hyperbolic bonus for exact fits, and a segmented function that penalizes small residuals while rewarding larger usable residuals with adaptive thresholds based on bin capacity and item size.

import numpy as np


def score(item, bins):
    remaining_after_assignment = bins - item
    utilization = 1 - (remaining_after_assignment / bins)
    future_potential = 2 / ((1 / (remaining_after_assignment + 1e-10)) + (1 / item))
    large_space_penalty = -np.where(remaining_after_assignment > item, remaining_after_assignment / bins, 0)
    full_bin_bonus = np.where(remaining_after_assignment == 0, 1000, 0)

    scores = utilization * future_potential + large_space_penalty + full_bin_bonus

    return scores
