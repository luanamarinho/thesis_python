import numpy as np
def generate_combinations(perplexity, early_exagg_range = (4, 32), theta_range = (0, 1.0)):
    """
    Generate combinations of parameters based on specified ranges.

    Parameters:
    - early_exagg_range: Tuple containing the start and stop values for early_exagg range.
    - theta_range: Tuple containing the start and stop values for theta range.
    - perplexity: single integer for the algorithm perplexity. Should be one of [5, 25, 45, 65, 90]

    Returns:
    - A list of parameter combinations.
    """
    
    if perplexity not in [5, 25, 45, 65, 90]:
        raise ValueError("Perplexity must be one of [5, 25, 45, 65, 90]")

    perplexity_values = [perplexity]
    early_exagg_values = np.linspace(*early_exagg_range, num=5, dtype=int).tolist()
    theta_values = np.round(np.linspace(*theta_range, num=5, dtype=float), 2).tolist()
    initial_momentum_values = [0.1, 0.3, 0.5]
    final_momentum_values = [0.8, 0.9, 1.0]

    # Generate combinations
    combinations = []
    for perplexity in perplexity_values:
        for early_exagg in early_exagg_values:
            for theta in theta_values:
                for initial_momentum in initial_momentum_values:
                    for final_momentum in final_momentum_values:
                        combinations.append((perplexity, early_exagg, initial_momentum, final_momentum, theta))

    return combinations

