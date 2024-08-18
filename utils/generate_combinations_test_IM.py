import numpy as np
from joblib import dump

def generate_combinations(num_combinations=50, split_value=0.283):
    """
    Generate 50 combinations of parameters with specific constraints.
    
    Returns:
    - A list of parameter combinations.
    """
    np.random.seed(42)
    
    # Perplexity, Early Exaggeration, Final Momentum, Theta are constant
    perplexity = 50
    early_exagg = 30
    final_momentum = 0.9
    theta = 0.5
    
    # Generate Initial Momentum values
    initial_momentum_lower = np.random.uniform(0.1, split_value, num_combinations//2)
    initial_momentum_upper = np.random.uniform(split_value, 0.5, num_combinations//2)
    
    initial_momentum_values = np.concatenate([initial_momentum_lower, initial_momentum_upper])
    np.random.shuffle(initial_momentum_values)  # Shuffle to mix lower and upper values
    
    # Create combinations
    combinations = [(perplexity, early_exagg, im, final_momentum, theta) for im in initial_momentum_values]
    
    return combinations

# Generate the combinations
combinations = generate_combinations()


# Display the first few combinations
#for i, combo in enumerate(combinations[:10], 1):
    #print(f"Combination {i}: Perplexity={combo[0]}, Early Exaggeration={combo[1]}, Initial Momentum={combo[2]:.3f}, Final Momentum={combo[3]}, Theta={combo[4]}")

# Convert the combinations to a DataFrame
#df = pd.DataFrame(combinations, columns=['Perplexity', 'Early_Exaggeration', 'Initial_Momentum', 'Final_Momentum', 'Theta'])
#sum(df['Initial_Momentum']<=0.28)

# Export
dump(combinations, 'output/tsne_parameter_combinations_IM_test.joblib')
#df.to_csv('output/tsne_parameter_combinations_IM_test.csv', index=False)

# Compute the correlation matrix
#correlation_matrix = df.corr()
