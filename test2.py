from scipy.io import mmread
from sklearn.decomposition import PCA
import pandas as pd
import scipy.sparse as sp
import numpy as np
import random
import time
import scanpy as sc
from anndata import AnnData
import sys
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
import itertools
import openTSNE
from openTSNE.affinity import PerplexityBasedNN
from joblib import dump, load


# Load data from .mtx file as a sparse matrix
data_sparse = mmread("data/matrix.mtx")
data_sparse.shape #(33694, 93575)
print(data_sparse.__class__) 
print(data_sparse.data.__class__)
print(data_sparse.data.shape)     #(114212920,)
print(data_sparse.data[:49])
#data_sparse_csr = data_sparse.tocsr()
#print(data_sparse_csr.__class__)


# Sampling to maximize representativeness
# 2000 7000 15000 30000 50000
def sampled_ind_matrix(metadata , nbr_samples = 2000):
    # Calculate the proportion of observations for each unique combination
    metadata_subset = metadata[['PatientNumber', 'TumorType', 'CellType']]
    group_proportions = metadata_subset.groupby(['PatientNumber', 'TumorType', 'CellType']).size() / len(metadata_subset)
    
    # Calculate the number of rows to sample from each unique combination
    sampled_rows_per_group = (group_proportions * nbr_samples).astype(int).clip(lower=1)

    def sample_rows(group, count):
        return group.sample(n=count, replace=False).index.tolist()
    
    # Apply the sampling function to each group and convert the result to a list of indices
    sampled_indices = metadata_subset.groupby(['PatientNumber', 'TumorType', 'CellType']).apply(lambda x: sample_rows(x, count=sampled_rows_per_group[x.name])).tolist()

    # Flatten the list of lists and limit it to the desired number of indices
    sampled_indices = [index for sublist in sampled_indices for index in sublist][:nbr_samples]

    return sampled_indices

metadata = pd.read_csv("data/2097-Lungcancer_metadata.csv.gz")
random.seed(1234)
ind_sampled_matrix = sampled_ind_matrix(metadata)

# Slicing metadata and data_sparce over the rows
metadata_sampled = metadata.iloc[ind_sampled_matrix, ]
metadata_sampled.duplicated(subset=['Cell'], keep=False).sum()
data_sparse_csr = data_sparse.transpose().tocsr()
data_sparse_csr = data_sparse_csr[ind_sampled_matrix]

# Find HVG
sys.getsizeof(AnnData(X = data_sparse_csr, obs = metadata_sampled)) #32595638
sys.getsizeof(AnnData(X = data_sparse_csr)) #32086102
adata = AnnData(X = data_sparse_csr)
sc.pp.highly_variable_genes(adata, n_top_genes=int(0.1 * adata.shape[1]), flavor='seurat_v3')
adata.var.sum()

def slice_adata_X_by_var(adata):
    highly_variable_genes = adata.var['highly_variable']
    sliced_X = adata.X[:, highly_variable_genes]
    return sliced_X

data_sp_csr_HVG = slice_adata_X_by_var(adata)

# Pre-processing: Normalization, Log-transformation, and scaling
def preprocess_sparse_matrix(data_sp_csr_HVG):
    """
    Preprocesses a sparse matrix by normalizing, log-transforming, and scaling its values.
    
    Parameters:
        data_sparse (scipy.sparse.csr_matrix): Input sparse matrix to be preprocessed.
        
    Returns:
        numpy.ndarray: Preprocessed dense matrix.
    """
    data_dense = data_sp_csr_HVG.toarray()
    data_normalized = normalize(data_dense, norm='l2', axis=1)

    # Log transform the data
    data_log_transformed = np.log1p(data_normalized)  # Apply log(1+x) transformation to avoid log(0)

    # Scale the data (z-score normalization)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log_transformed)

    return data_scaled

dense_data_scaled = preprocess_sparse_matrix(data_sp_csr_HVG)
np.save('dense_data_scaled.npy', dense_data_scaled)
dense_data_scaled = np.load('dense_data_scaled.npy')
np.savez('dense_data_scaled_compressed', dense_data_scaled_compressed = dense_data_scaled)

f = gzip.GzipFile("dense_data_scaled_compressed_gzip", "w")
np.save(file=f,  arr = dense_data_scaled)
f.close()
f = gzip.GzipFile('dense_data_scaled_compressed_gzip', "r"); a = np.load(f)
#dense_data_scaled[:10]
#np.shape(dense_data_scaled) #(1978, 3369)


# Perform PCA on pre-processed, downsampled dense matrix
pca = PCA(n_components=50, svd_solver = 'randomized', random_state = 1234)
#start_time = time.time()
data_pca = pca.fit_transform(dense_data_scaled)
#end_time = time.time()
#runtime = end_time - start_time #0.7032742500305176
#np.shape(data_pca) #(1978, 50)

# TSNE
# Prof Aerts: https://opentsne.readthedocs.io/en/stable/benchmarks.html
#https://opentsne.readthedocs.io/en/stable/api/index.html#openTSNE.TSNE.prepare_initial
#https://opentsne.readthedocs.io/en/stable/api/index.html#openTSNE.TSNE.fit

# Parameters:
#perplexity = np.linspace(start = 5, stop = 90, num=18, dtype=int).tolist() # default 30 included
#early_exaggeration = np.linspace(start = 4, stop = 32, num=15, dtype=int).tolist() # default 12 included
#exaggageration The exaggeration factor to use during the normal optimization phase. This can be used to form more densely packed clusters and is useful for large data sets
# The late exaggeration technique is implemented in a variant of t-SNE called "Multicore t-SNE," 
#early_exaggeration_iter. The number of iterations to run in the early exaggeration phase. Default = 250
#initial_momentum = np.round(np.linspace(start = 0.1, stop = 0.5, num=5, dtype=float), 1).tolist() # default in the pkg is 0.8, but 0.5 in R scatter (included)
#final_momentum = np.round(np.linspace(start = 0.8, stop = 1.0, num=8, dtype=float), 2).tolist() # default = 0.8 included
#theta = np.round(np.linspace(start = 0, stop = 1.0, num=11, dtype=float), 2).tolist() # defaults 0 and 0.5 included
#dof default = 1
#n_iter (int) – The number of iterations to run in the normal optimization regime. Default 500
initialization = 'random' # default pca, but...
negative_gradient_method = 'BH' # For larger data sets, FFT is recommended
random_state = 1234

def generate_combinations(perplexity_range, early_exagg_range, initial_momentum_range, 
                          final_momentum_range, theta_range):
    """
    Generate combinations of parameters based on specified ranges.

    Parameters:
    - perplexity_range: Tuple containing the start and stop values for perplexity range.
    - early_exagg_range: Tuple containing the start and stop values for early_exagg range.
    - initial_momentum_range: Tuple containing the start and stop values for initial_momentum range.
    - final_momentum_range: Tuple containing the start and stop values for final_momentum range.
    - theta_range: Tuple containing the start and stop values for theta range.

    Returns:
    - A list of parameter combinations.
    """
    # Generate parameter values using np.linspace
    perplexity_values = np.linspace(*perplexity_range, num=18, dtype=int).tolist()
    early_exagg_values = np.linspace(*early_exagg_range, num=15, dtype=int).tolist()
    initial_momentum_values = np.round(np.linspace(*initial_momentum_range, num=5, dtype=float), 1).tolist()
    final_momentum_values = np.round(np.linspace(*final_momentum_range, num=8, dtype=float), 2).tolist()
    theta_values = np.round(np.linspace(*theta_range, num=11, dtype=float), 2).tolist()

    # Generate combinations
    combinations = list(itertools.product(perplexity_values, early_exagg_values, 
                                          initial_momentum_values, final_momentum_values, theta_values))

    return combinations

# Define a cache dictionary to store computed affinities
affinity_cache = {}
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
    for perplexity in perplexity_values:
        # Check if affinity for this perplexity has been computed
        if perplexity not in affinity_cache:
            # Compute affinity and store in cache
            affinities = PerplexityBasedNN(X, perplexity=perplexity, n_jobs=n_jobs, random_state=random_state)
            affinity_cache[perplexity] = affinities

    return affinity_cache

affinity_cache = compute_affinities(X = dense_data_scaled, perplexity_values = np.linspace(*(5, 90), num=18, dtype=int).tolist())
dump(affinity_cache, 'affinity_cache.joblib')

def run_openTSNE_with_combinations(combinations, X, affinity_cache, initialization='random', n_jobs=1, 
                                   negative_gradient_method='BH', random_state=1234, 
                                   n_iter=750, verbose=False, dof=1):
    """
    Run openTSNE with a list of parameter combinations.

    Parameters:
    - combinations: List of parameter combinations.
    - X: Input data matrix.
    - affinity_cache: Dictionary containing pre-computed affinities.
    - Other parameters as in the previous function.

    Returns:
    - List of tuples, each containing the combination and the resulting embedding.
    """
    results = []

    for combo in combinations:
        perplexity, early_exagg, initial_momentum, final_momentum, theta = combo

        # Get pre-computed affinities from cache
        affinities = affinity_cache[perplexity]

        # Create TSNE object with the provided parameters
        tsne = openTSNE.TSNE(
            perplexity=perplexity,
            early_exaggeration=early_exagg,
            initialization=initialization,
            n_jobs=n_jobs,
            negative_gradient_method=negative_gradient_method,
            theta=theta,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state,
            dof=dof
        )

        start_time = time.time()

        # Fit TSNE with cached affinities
        embedding = tsne.fit(affinities=affinities)

        runtime = time.time() - start_time

        KL_divergence = embedding.kl_divergence
        
        # Append the results
        results.append((combo, embedding, runtime, KL_divergence))

    return results   

# Define parameter ranges
perplexity_range = (5, 90)          # perplexity range
early_exagg_range = (4, 32)          # early_exagg range
initial_momentum_range = (0.1, 0.5)  # initial_momentum range
final_momentum_range = (0.8, 1.0)    # final_momentum range
theta_range = (0, 1.0)              # theta range

# Generate combinations
combinations = generate_combinations(perplexity_range, early_exagg_range, initial_momentum_range, 
                                     final_momentum_range, theta_range)

# Run TSNE with the generated combinations
combination = [(50, 4, 0.1, 0.8, 0.8)] #[(10, 4, 0.1, 0.8, 0.8)]
results = run_openTSNE_with_combinations(combination, X = dense_data_scaled, affinity_cache = affinity_cache, verbose=True) #  Time elapsed: 388.99 seconds early phase | Time elapsed: 1257.42 seconds
for combo, embedding, runtime, KL_divergence  in results:
    print("Parameter combination:", combo)
    print("Embedding shape:", embedding.shape)
    print("Embedding:", embedding.__class__)
    print("Embedding:", embedding)
    print("Runtime:", runtime)
    print("KL divergence:", [KL_divergence])
    
# Unpack the results
#dump(results, 'results.joblib')
import matplotlib.pyplot as plt
# Unpack the single tuple from results
embedding_X, embedding_Y = results[0][1][:, 0], results[0][1][:, 1]
runtime = results[0][2]
KL_divergence =  results[0][3]
combined_data = pd.DataFrame({
    'TSNE_1': embedding_X,
    'TSNE_2': embedding_Y,
    'CellType': metadata_sampled['CellType']
})
combined_data['CellType'] = pd.Categorical(combined_data['CellType'])

# Plot using DataFrame plot method
combined_data.plot.scatter(x='TSNE_1', y='TSNE_2', c='CellType', cmap = 'tab10')
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.title('TSNE Plot for combination [(50, 4, 0.1, 0.8, 0.8)]')
plt.show()

dump([combined_data, runtime, KL_divergence], 'combined_data[(50, 4, 0.1, 0.8, 0.8)].joblib')
