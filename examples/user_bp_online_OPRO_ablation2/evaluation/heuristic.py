# Second best heuristic extracted from training results
# Objective: 0.00624
# Algorithm: This algorithm calculates scores by combining a bin's remaining capacity with a Gaussian-based penalty for new bins, an exponential bonus for utilization, a hyperbolic bonus for exact fits, and a segmented function that penalizes small residuals while rewarding larger usable residuals with adaptive thresholds based on bin capacity and item size.

import numpy as np


def score(item, bins):
    remaining_after_assignment = bins - item
    normalized_remaining = remaining_after_assignment / bins.max()
    underutilization_penalty = -np.exp(-remaining_after_assignment / (1.5 * item + 1e-10)) * (remaining_after_assignment > 0)
    fill_ratio_bonus = np.sqrt((bins - remaining_after_assignment) / bins)
    fragmentation_penalty = -np.sqrt(remaining_after_assignment / bins.max()) * (remaining_after_assignment > 0)
    scores = normalized_remaining + underutilization_penalty + fill_ratio_bonus + fragmentation_penalty
    scores[bins == bins.max()] = -np.inf
    return scores
