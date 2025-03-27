import numpy as np
def generate_combinations(perplexity_range = (5, 150),
                          early_exagg_range = (4, 32),
                          theta_range = (0, 1.0),
                          initial_momem_range = (0.1, 0.5),
                          final_momem_range = (0.8, 1),
                          num_combinations = 600,
                          seed = 42):
    """
    Generate combinations of parameters based on specified ranges.

    Parameters:
    - perplexity_range: Tuple containing the start and stop values for perplexity range.
    - early_exagg_range: Tuple containing the start and stop values for early_exagg range.
    - theta_range: Tuple containing the start and stop values for theta range.
    - initial_momem_range: Tuple containing the start and stop values for initial_momem range.
    - final_momem_range: Tuple containing the start and stop values for final_momem range.
    - num_combinations: Number of combinations to generate.
    - seed: Seed for the random number generator.

    Returns:
    - A list of tuples with random combinations of the 5 Barnes-Hut-SNE parameters.
    """

    np.random.seed(seed)

    perplexity_values = np.linspace(*perplexity_range, num=num_combinations//5).tolist()
    early_exagg_values = np.linspace(*early_exagg_range, num=num_combinations//5).tolist()
    theta_values = np.round(np.linspace(*theta_range, num=num_combinations//5, dtype=float), 2).tolist()
    initial_momentum_values = np.linspace(*initial_momem_range, num=num_combinations//5).tolist()
    final_momentum_values = np.linspace(*final_momem_range, num=num_combinations//5).tolist()

    combinations = set()
    
    # Generate combinations until we have the desired number
    while len(combinations) < num_combinations:
        perplexity = np.random.choice(perplexity_values)
        early_exagg = np.random.choice(early_exagg_values)
        theta = np.random.choice(theta_values)
        initial_momentum = np.random.choice(initial_momentum_values)
        final_momentum = np.random.choice(final_momentum_values)
        combinations.add((perplexity, early_exagg, initial_momentum, final_momentum, theta))

    return list(combinations)
