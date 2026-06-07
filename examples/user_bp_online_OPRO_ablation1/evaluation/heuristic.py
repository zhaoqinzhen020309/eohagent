# Second best heuristic extracted from training results
# Objective: 0.00624
# Algorithm: This algorithm calculates scores by combining a bin's remaining capacity with a Gaussian-based penalty for new bins, an exponential bonus for utilization, a hyperbolic bonus for exact fits, and a segmented function that penalizes small residuals while rewarding larger usable residuals with adaptive thresholds based on bin capacity and item size.

import numpy as np


def score(item, bins):
    remaining_after_assignment = bins - item
    fill_ratio = 1 - (remaining_after_assignment / bins)
    future_potential_penalty = -np.where(remaining_after_assignment > item, 1.0 / (remaining_after_assignment + 1e-10),
                                         0)
    waste_penalty = -np.where(remaining_after_assignment > item, 1.0 / (remaining_after_assignment + 1e-10), 0)
    large_space_penalty = -np.where(bins > item, 1.0 / (bins + 1e-10), 0)
    exact_fit_bonus = np.where(remaining_after_assignment == 0, 1000, 0)

    weight = 0.3 + 0.7 * fill_ratio
    scores = (1 - weight) * future_potential_penalty + weight * waste_penalty + large_space_penalty + exact_fit_bonus

    scores[bins == bins.max() - item] = -np.inf  # Penalize bins that are almost full to avoid fragmentation

    return scores
