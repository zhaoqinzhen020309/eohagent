# Second best heuristic extracted from training results
# Objective: 0.00624
# Algorithm: This algorithm calculates scores by combining a bin's remaining capacity with a Gaussian-based penalty for new bins, an exponential bonus for utilization, a hyperbolic bonus for exact fits, and a segmented function that penalizes small residuals while rewarding larger usable residuals with adaptive thresholds based on bin capacity and item size.

import numpy as np


def score(item, bins):
    avg_bin_capacity = np.mean(bins)
    remaining = bins - item
    normalized_remaining = remaining / avg_bin_capacity
    low_remaining_penalty = np.where(remaining <= 0.2 * avg_bin_capacity, -1e9, 0)
    exact_fit_bonus = np.where(remaining == 0, 5000, 0)
    dynamic_weight = 0.3 + 0.7 * (item / avg_bin_capacity)
    scores = (1 - dynamic_weight) * normalized_remaining + dynamic_weight * (1 / (remaining + 1e-10)) + exact_fit_bonus + low_remaining_penalty
    scores[bins == bins.max()] = -np.inf
    return scores
