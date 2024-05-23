import numpy as np
import itertools
import pandas as pd


def generate_combinations(early_exagg_range, initial_momentum_range, 
                          final_momentum_range, theta_range):
    """
    Generate combinations of parameters based on specified ranges.

    Parameters:
    - early_exagg_range: Tuple containing the start and stop values for early_exagg range.
    - initial_momentum_range: Tuple containing the start and stop values for initial_momentum range.
    - final_momentum_range: Tuple containing the start and stop values for final_momentum range.
    - theta_range: Tuple containing the start and stop values for theta range.

    Returns:
    - A list of parameter combinations.
    """
    # Generate parameter values using np.linspace
    perplexity_values = [5, 25, 45, 65, 90]
    early_exagg_values = np.linspace(*early_exagg_range, num=5, dtype=int).tolist()
    initial_momentum_values = [0.1, 0.3, 0.5]
    #initial_momentum_values = np.round(np.linspace(*initial_momentum_range, num=5, dtype=float), 1).tolist()
    final_momentum_values = [0.8, 0.9, 1.0]
    #final_momentum_values = np.round(np.linspace(*final_momentum_range, num=5, dtype=float), 2).tolist()
    theta_values = np.round(np.linspace(*theta_range, num=5, dtype=float), 2).tolist()

    # Generate combinations
    combinations = list(itertools.product(perplexity_values, early_exagg_values, 
                                          initial_momentum_values, final_momentum_values, theta_values))

    return combinations


# Define parameter ranges
perplexity_range = (5, 90)          # perplexity range
early_exagg_range = (4, 32)          # early_exagg range
initial_momentum_range = (0.1, 0.5)  # initial_momentum range
final_momentum_range = (0.8, 1.0)    # final_momentum range
theta_range = (0, 1.0)              # theta range


# Generate combinations
combinations = generate_combinations(early_exagg_range, initial_momentum_range, 
                                     final_momentum_range, theta_range)
len(combinations) #3125 #1125


filtered_tuples = [tup for tup in combinations if tup[0] == 5]
np.random.seed(1234)
ind = np.random.choice(len(filtered_tuples), size=80, replace=False)
filtered_tuples_array = np.array(filtered_tuples)
filtered_tuples_sampled = filtered_tuples_array[ind]


combinations_df_2 = pd.DataFrame(combinations,

                               columns=['Perplexity',
                                        'Early_exaggeration',
                                        'Initial_momentum',
                                        'Final_momentum',
                                        'Theta'])
combinations_df_2 = combinations_df_2.sort_values(['Perplexity', 'Early_exaggeration', 'Theta'])
combinations_df_2[:9]
combinations_df_2[9:18]
combinations_df_2.to_csv('combinations_df_2', index=False)
combinations_perp5_df = combinations_df_2.loc[combinations_df_2['Perplexity']==5]

# Transform DataFrame to list of tuples
combinations_perp5 = [tuple(row) for row in combinations_perp5_df.itertuples(index=False)]

print(combinations_perp5)


import numpy as np

# Sample size per group
sample_size = 4

def sample_rows(group, count):
    return group.sample(n=min(count, len(group)), replace=False).index.tolist()

group_proportions = combinations_df.groupby(['Perplexity', 'Early_exaggeration', 'Theta']).size()

sampled_rows_per_group = (group_proportions).astype(int).clip(lower=1)

np.random.seed(1234)

sampled_indices = combinations_df.groupby(['Perplexity', 'Early_exaggeration', 'Theta']).apply(lambda x: sample_rows(x, count=sample_size)).apply(list)

sampled_indices = [index for sublist in sampled_indices for index in sublist]

print(sampled_indices)

len(sampled_indices)

combinations_df_sampled = combinations_df.iloc[sampled_indices]


######--------------------

def generate_combinations(early_exagg_range, theta_range):
    """
    Generate combinations of parameters based on specified ranges.

    Parameters:
    - early_exagg_range: Tuple containing the start and stop values for early_exagg range.
    - theta_range: Tuple containing the start and stop values for theta range.

    Returns:
    - A list of parameter combinations.
    """
    # Generate parameter values using np.linspace
    perplexity_values = [5, 25, 45, 65, 90]
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

# Define parameter ranges
early_exagg_range = (4, 32)          # early_exagg range
theta_range = (0, 1.0)

# Generate combinations
combinations = generate_combinations(early_exagg_range, theta_range)

# Create DataFrame
combinations_df = pd.DataFrame(combinations, columns=['Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta'])
