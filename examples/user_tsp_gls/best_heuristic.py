import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    
    mean_usage = np.mean(edge_n_used)
    usage_std = np.std(edge_n_used)
    global_scale = 1.0 + 0.6 * np.sqrt(usage_std / (mean_usage + 1e-8))
    
    max_usage = np.max(edge_n_used)
    min_usage = np.min(edge_n_used)
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        normalized_freq = (edge_n_used[u, v] - min_usage) / (max_usage - min_usage + 1e-8)
        usage_deviation = (edge_n_used[u, v] - mean_usage) / (usage_std + 1e-8)
        
        sigmoid_factor = 1.0 + 0.4 / (1.0 + np.exp(-usage_deviation * 1.2))
        sqrt_inverse_freq = np.sqrt(1.0 / (normalized_freq + 0.15))
        penalty = global_scale * sigmoid_factor * np.exp(0.8 * sqrt_inverse_freq)
        
        updated_edge_distance[u, v] *= penalty
        updated_edge_distance[v, u] *= penalty
    
    return updated_edge_distance