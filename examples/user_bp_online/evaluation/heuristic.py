# Second best heuristic extracted from training results
# Objective: 0.00624
# Algorithm: This algorithm calculates scores by combining a bin's remaining capacity with a Gaussian-based penalty for new bins, an exponential bonus for utilization, a hyperbolic bonus for exact fits, and a segmented function that penalizes small residuals while rewarding larger usable residuals with adaptive thresholds based on bin capacity and item size.

import numpy as np


def score(item, bins):
    valid_bins = bins > item
    scores = np.zeros_like(bins)
    if not np.any(valid_bins):
        return scores
    exp_capacity = np.exp(-0.1 * (item / bins[valid_bins]))
    num_items = len(bins) - np.arange(len(bins))
    adjusted_num_items = num_items[valid_bins]
    cubic_penalty = 0.1 * adjusted_num_items ** 3
    if len(bins[valid_bins]) > 0:
        median_item_size = np.median(bins[valid_bins] - item)
        even_distribution_bonus = -np.abs((bins[valid_bins] - item) - median_item_size) ** 2
    else:
        even_distribution_bonus = 0
    scores[valid_bins] = exp_capacity - cubic_penalty + even_distribution_bonus

    return scores