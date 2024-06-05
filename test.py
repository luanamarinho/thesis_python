import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
from sklearn.manifold import trustworthiness

def compute_distance_matrix_chunked(data, n_jobs=1):
    n_samples = data.shape[0]
    full_distance_matrix = np.zeros((n_samples, n_samples))

    start_idx = 0
    for chunk in pairwise_distances_chunked(data, n_jobs=n_jobs):
        end_idx = start_idx + chunk.shape[0]
        full_distance_matrix[start_idx:end_idx] = chunk
        start_idx = end_idx

    return full_distance_matrix

def compute_knn_indices(distance_matrix, rm_1st_column = False):
    # Get the indices of the k-nearest neighbors using np.argsort
    indices = np.argsort(distance_matrix, axis=1)
    if rm_1st_column:
        indices = indices[:, 1:]
    return indices

def compute_trustworthiness_continuity(tsne_configuration, original_data, k, n_jobs=2):
    # Compute the trustworthiness
    trust = trustworthiness(original_data, tsne_configuration, n_neighbors=k)

    # Compute the pairwise distance matrix for the original data in chunks
    original_distance_matrix = compute_distance_matrix_chunked(original_data, n_jobs=n_jobs)

    # Get the k-nearest neighbors indices for the original data
    original_knn_indices = compute_knn_indices(original_distance_matrix, k)

    # Compute the pairwise distance matrix for the t-SNE data in chunks
    tsne_distance_matrix = compute_distance_matrix_chunked(tsne_configuration, n_jobs=n_jobs)

    # Get the k-nearest neighbors indices for the t-SNE data
    tsne_knn_indices = compute_knn_indices(tsne_distance_matrix, k)

    # Compute continuity
    n_samples = original_data.shape[0]
    continuity = 0
    for i in range(n_samples):
        continuity += len(set(tsne_knn_indices[i]).intersection(set(original_knn_indices[i])))
    continuity /= (n_samples * k)
    
    return trust, continuity

def main(tsne_filepath, original_data_filepath, k=10, n_jobs=2):
    # Load the t-SNE configuration
    df_tsne = pd.read_csv(tsne_filepath, compression='gzip')
    tsne_configuration = df_tsne.iloc[:, :2].values
    
    # Load the original high-dimensional data
    df_original = pd.read_csv(original_data_filepath, compression='gzip')
    original_data = df_original.values

    # Compute the trustworthiness and continuity
    trust, continuity = compute_trustworthiness_continuity(tsne_configuration, original_data, k, n_jobs)
    
    print(f"Trustworthiness: {trust}")
    print(f"Continuity: {continuity}")

# Example usage
main('output/df_tsne_unique.csv', 'output/original_data.csv', k=10, n_jobs=2)
