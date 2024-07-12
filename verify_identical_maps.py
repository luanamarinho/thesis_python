import pandas as pd
import numpy as np
import hashlib

# Create a dummy DataFrame with random data
df_tsne = pd.read_csv('output/df_tsne_unique.csv', compression='gzip')

# Correct reshaping: Ensure each pair of columns is treated as a single map
num_rows = df_tsne.shape[0]
num_maps = df_tsne.shape[1] // 2

# Create a list to store each map as a separate row
maps_list = []

for i in range(num_maps):
  # Extract each pair of columns and flatten them into a 1D array
  map_pair = df_tsne.iloc[:, 2*i:2*i+2].values.flatten()
  maps_list.append(map_pair)

# Convert the list of maps to a NumPy array
maps_array = np.array(maps_list)

# Define a function to hash each row (map)
def hash_array(array):
    array_str = ','.join(map(str, array))
    return hashlib.md5(array_str.encode()).hexdigest()

# Apply the hashing function to each map and store hashes
hashes = np.array([hash_array(map_row) for map_row in maps_array])

# Identify unique maps by finding unique hashes
_, unique_indices = np.unique(hashes, return_index=True)

# Get the unique maps
unique_maps = maps_array[unique_indices]


## Compare with previous indices
# Load data
df_parameters_results = pd.read_csv('output/df_parameters_results.csv')
print(df_parameters_results.columns)
parameters = ['Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta']
outcomes = ['KL_divergence', 'trust_k30', 'trust_k300', 'stress']

unique_stress, ind_unique_stress = np.unique(df_parameters_results['stress'], return_index=True)
print(f'Number of unique values of shepard stress: [{len(unique_stress)}]')
print(sorted(ind_unique_stress)[:10])

print(f'Indices of unique shepard stress and unique maps are identical: [{sorted(ind_unique_stress) == sorted(unique_indices)}]')

dupli_indices = np.setdiff1d(np.arange(0, num_maps), unique_indices)

combinations_unique20 = sorted(np.random.RandomState(42).choice(unique_indices, 20, replace=False))
combinations_replications30 = sorted(np.random.RandomState(42).choice(dupli_indices, 30, replace=False))

combined_indices = np.concatenate([combinations_unique20, combinations_replications30])
col_keep = parameters

# Subset the DataFrame
df_combinations_verify = df_parameters_results.iloc[combined_indices][col_keep]

labels = ['unique'] * 20 + ['replication'] * 30
df_combinations_verify['Type'] = labels

df_combinations_verify.to_csv('output/df_combinations_verify.csv', index=True)
#imported_df = pd.read_csv('output/df_combinations_verify.csv', index_col=0)




df = df_parameters_results[['Perplexity', 'Early_exaggeration', 'Theta']]
unique_combinations = df.drop_duplicates()
len(unique_combinations)

df2 = df_parameters_results[['Perplexity', 'Early_exaggeration', 'Theta', 'KL_divergence']]
unique_combinations2 = df2.drop_duplicates()
len(unique_combinations2)

## in R
#df = read.csv('output/df_parameters_results.csv')
#df = df[, c('Perplexity', 'Early_exaggeration', 'Theta', 'KL_divergence')]

#grouped = df |> dplyr::group_by(Perplexity, Early_exaggeration, Theta, KL_divergence) |>
#  dplyr::summarize(count = dplyr::n())

#View(grouped)
#> dim(grouped)
#[1] 93  5
#Momentums were not passed!