import numpy as np
def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    # Step 1: Calculate machine completion times for current sequence
    completion_times = np.zeros((n, m))
    for i, job in enumerate(current_sequence):
        for machine in range(m):
            if i == 0 and machine == 0:
                completion_times[i, machine] = time_matrix[job, machine]
            elif i == 0:
                completion_times[i, machine] = completion_times[i, machine-1] + time_matrix[job, machine]
            elif machine == 0:
                completion_times[i, machine] = completion_times[i-1, machine] + time_matrix[job, machine]
            else:
                completion_times[i, machine] = max(completion_times[i-1, machine], completion_times[i, machine-1]) + time_matrix[job, machine]
    
    # Step 2: Identify critical path and calculate job criticality
    job_criticality = np.zeros(n)
    for i in range(n-1, -1, -1):
        job_idx = current_sequence[i]
        if i == n-1:
            job_criticality[job_idx] = completion_times[i, m-1]
        else:
            next_job_crit = job_criticality[current_sequence[i+1]]
            machine_delays = completion_times[i, :] - (completion_times[i+1, :] - time_matrix[current_sequence[i+1], :])
            job_criticality[job_idx] = np.max(machine_delays) + next_job_crit
    
    # Step 3: Calculate machine load imbalance
    machine_loads = np.sum(completion_times, axis=0)
    mean_load = np.mean(machine_loads)
    load_imbalance = np.abs(machine_loads - mean_load)
    
    # Step 4: Combine criticality and load imbalance for perturbation selection using sigmoid scaling
    combined_metric = job_criticality * (1 + load_imbalance[np.argmax(completion_times[current_sequence], axis=1)])
    sigmoid_scaled_factor = 1 / (1 + np.exp(-combined_metric / np.max(combined_metric)))
    
    # Step 5: Update time matrix with adaptive perturbation based on sigmoid scaled factor using ant colony optimization principles
    pheromone_trails = np.random.rand(n, m)
    new_matrix = time_matrix.copy()
    for job in range(n):
        perturbation_factor = 1.0 + 0.2 * (sigmoid_scaled_factor[job] - 0.5) * np.random.randn()
        new_matrix[job, :] *= max(0.1, perturbation_factor)
        new_matrix[job, :] += pheromone_trails[job, :]
    
    # Step 6: Select top jobs for perturbation using roulette wheel selection based on combined criticality and load imbalance
    weights = sigmoid_scaled_factor / np.sum(sigmoid_scaled_factor)
    k = max(2, int(0.2 * n))
    perturb_jobs = np.random.choice(n, size=k, replace=False, p=weights)

    return new_matrix.astype(np.float64), perturb_jobs

    return new_matrix