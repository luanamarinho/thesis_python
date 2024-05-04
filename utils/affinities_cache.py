from openTSNE.affinity import PerplexityBasedNN
def compute_affinities(X, perplexity_values, n_jobs=1, random_state=1234):
    """
    Compute affinities for multiple perplexity values and cache them.

    Parameters:
    - X: Input data matrix.
    - perplexity_values: List of perplexity values.
    - n_jobs: Number of parallel jobs.
    - random_state: Random seed for reproducibility.

    Returns:
    - A dictionary containing affinities computed for each perplexity value.
    """
    affinity_cache ={}
    for perplexity in perplexity_values:
        # Check if affinity for this perplexity has been computed
        if perplexity not in affinity_cache:
            # Compute affinity and store in cache
            affinities = PerplexityBasedNN(X, perplexity=perplexity, n_jobs=n_jobs, random_state=random_state)
            affinity_cache[perplexity] = affinities

    return affinity_cache