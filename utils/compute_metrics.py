import numpy as np
import pandas as pd
from joblib import load
from utils.compute_distances import compute_distance_matrix_chunked
from utils.trustworthiness import trustworthiness_continuity

def compute_metrics(df_tsne_filepath, dist_X_filepath, chunk_size=1000, min_k=1, max_k=50):
    # Load input pairwise distance matrix
    dist_X = load(dist_X_filepath)
    n_samples = dist_X.shape[0]
    
    # Initialize variables to store trustworthiness results
    T_k_aggregate = np.zeros(max_k - min_k + 1)

    # Load t-SNE maps
    df_tsne = pd.read_csv(df_tsne_filepath, compression='gzip')
    df_tsne = df_tsne.iloc[:, 0:2]

    # Loop through chunks
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        
        # Compute pairwise distances for the chunk
        dist_embedded_chunk = compute_distance_matrix_chunked(data=df_tsne.iloc[start:end])

        # Compute trustworthiness for the chunk
        T_k_chunk = trustworthiness_continuity(dist_X[start:end, start:end], dist_embedded_chunk, min_k, max_k)
        
        # Aggregate trustworthiness results
        T_k_aggregate += np.array(T_k_chunk)

    # Normalize trustworthiness values
    for k in range(min_k, max_k + 1):
        T_k_aggregate[k - min_k] /= (2 / (n_samples * k * (2 * n_samples - 3 * k - 1)))

    # Compute final trustworthiness results
    T_k_final = 1 - T_k_aggregate

    return T_k_final
