import numpy as np
def generate_combinations(perplexity_range = (5, 150),
                          early_exagg_range = (4, 32),
                          theta_range = (0, 1.0),
                          initial_momem_range = (0.1, 0.5),
                          final_momem_range = (0.8, 1),
                          num_combinations = 600):
    """
    Generate combinations of parameters based on specified ranges.

    Parameters:
    - early_exagg_range: Tuple containing the start and stop values for early_exagg range.
    - theta_range: Tuple containing the start and stop values for theta range.
    - perplexity: single integer for the algorithm perplexity. Should be one of [5, 25, 45, 65, 90]

    Returns:
    - A list of parameter combinations.
    """
    
    #if perplexity not in [5, 25, 45, 65, 90]:
        #raise ValueError("Perplexity must be one of [5, 25, 45, 65, 90]")

    np.random.seed(42)

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


# Convert the combinations to a DataFrame
#df = pd.DataFrame(combinations, columns=['Perplexity', 'Early_Exaggeration', 'Initial_Momentum', 'Final_Momentum', 'Theta'])

# Export the DataFrame to a CSV file
#df.to_csv('output/tsne_parameter_combinations.csv', index=False)

# Compute the correlation matrix
#correlation_matrix = df.corr()