import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    machine_completion = np.zeros(m)
    job_completion = np.zeros(n)
    
    for job in current_sequence:
        start_time = 0
        for machine in range(m):
            machine_ready = machine_completion[machine]
            start_time = max(start_time, machine_ready)
            end_time = start_time + time_matrix[job, machine]
            machine_completion[machine] = end_time
            start_time = end_time
        job_completion[job] = end_time
    
    makespan = np.max(machine_completion)
    
    machine_criticality = np.zeros(m)
    for machine in range(m):
        critical_impact = 0
        for job in range(n):
            remaining_time = np.sum(time_matrix[job, machine:])
            critical_impact += remaining_time * (machine_completion[machine] / makespan)
        machine_criticality[machine] = critical_impact
    
    job_flexibility = np.zeros(n)
    for job in range(n):
        bottleneck_degree = np.max(time_matrix[job]) / np.sum(time_matrix[job])
        temporal_position = job_completion[job] / makespan
        critical_exposure = np.sum(time_matrix[job] * machine_criticality) / np.sum(time_matrix[job])
        
        job_flexibility[job] = 0.5 * bottleneck_degree + 0.3 * (1 - temporal_position) + 0.2 * critical_exposure
    
    new_matrix = time_matrix.copy()
    max_flexibility = np.max(job_flexibility) if np.max(job_flexibility) > 0 else 1
    
    for job in range(n):
        flexibility_ratio = job_flexibility[job] / max_flexibility
        perturbation = 1.0 + 0.25 * flexibility_ratio * (np.random.random() - 0.3)
        for machine in range(m):
            new_matrix[job, machine] = max(0.1, time_matrix[job, machine] * perturbation)
    
    k = max(2, min(5, n // 3))
    perturb_jobs = np.argsort(job_flexibility)[-k:]
    
    return new_matrix, perturb_jobs