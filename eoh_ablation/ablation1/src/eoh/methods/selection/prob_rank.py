import random

def parent_selection(pop, m):
    """Original parent selection for EoH operators"""
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = random.choices(pop, weights=probs, k=m)
    return parents

def ranked_selection(pop, num):
    """
    SEC V6: Select top-ranked individuals from population.
    Used for providing parent samples to LLM 2 (Strategist).
    
    Args:
        pop: Population list (each individual is a dict with 'objective' key)
        num: Number of individuals to select
    
    Returns:
        List of top num individuals, sorted by objective (ascending = better)
    """
    if not pop:
        return []
    
    # Filter out invalid individuals
    valid_pop = [ind for ind in pop if ind.get('objective') is not None]
    if not valid_pop:
        return []
    
    # Sort by objective (ascending - lower is better)
    sorted_pop = sorted(valid_pop, key=lambda x: x['objective'])
    
    # Return top num individuals
    return sorted_pop[:min(num, len(sorted_pop))]