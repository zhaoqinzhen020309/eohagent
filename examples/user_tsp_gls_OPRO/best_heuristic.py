import numpy as np
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = len(local_opt_tour)
    base_penalty_factor = 1.2
    usage_weight = 0.1
    decay_rate = 0.95
    sigmoid_scale = 10.0
    updated_edge_distance = edge_distance.copy()
    
    for i in range(n):
        from_node = local_opt_tour[i]
        to_node = local_opt_tour[(i + 1) % n]
        
        usage_count = edge_n_used[from_node, to_node] if edge_n_used[from_node, to_node] > 0 else 1
        linear_penalty = 1.0 + usage_weight * usage_count
        harmonic_scaling = n / (1 + usage_count)
        sigmoid_exploration = 1.0 / (1.0 + np.exp(-sigmoid_scale * usage_weight * usage_count))
        decay_factor = base_penalty_factor ** (usage_count ** -decay_rate)
        quadratic_penalty = 1.0 + 0.02 * usage_count ** 2
        exponential_decay = np.exp(-0.1 * usage_count)
        
        total_penalty = linear_penalty * harmonic_scaling * sigmoid_exploration * decay_factor * quadratic_penalty * exponential_decay
        
        updated_edge_distance[from_node, to_node] *= total_penalty
        updated_edge_distance[to_node, from_node] *= total_penalty
    
    return updated_edge_distance
