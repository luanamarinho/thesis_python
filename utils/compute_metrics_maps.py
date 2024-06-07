import numpy as np
import pandas as pd
from joblib import load
from utils.trustworthiness_chunks_1k import trustworthiness_chunks
from utils.compute_distances import compute_distance_matrix_chunked

def compute_metrics_maps(dist_input_filepath, df_tsne_filepath, chunk_size=1000, k=1):
    # Load input distance matrix
    X = load(dist_input_filepath).astype(np.float32)
    n_samples = X.shape[0]

    # Load t-SNE maps
    df_tsne = pd.read_csv(df_tsne_filepath, compression='gzip', dtype=np.float32)
    n_maps = 100 #df_tsne.shape[1] // 2  # Number of t-SNE maps (pairs of columns)

    # Initialize a list to store trustworthiness results for all maps
    trustworthiness_list = []

    for i in range(n_maps):
        # Extract the current t-SNE map
        df_tsne_map = df_tsne.iloc[:, 2*i:2*i+2].to_numpy()
        
        # Compute distances for the current t-SNE map
        X_embedded = compute_distance_matrix_chunked(df_tsne_map).astype(np.float32)

        # Initialize variables to store trustworthiness results for the current map
        T_k_aggregate = np.zeros(1, dtype=np.float32)

        # Split data into chunks
        #chunks = [(X[j:j+chunk_size], X_embedded[j:j+chunk_size]) for j in range(0, n_samples, chunk_size)]

        # Compute trustworthiness for each chunk
        #for chunk in chunks:
        for j in range(0, n_samples, chunk_size):
            # Compute trustworthiness for the chunk for each k
            X_chunk = X[j:j+chunk_size]
            X_embedded_chunk = X_embedded[j:j+chunk_size]
            T_k_chunk = trustworthiness_chunks(X_chunk, X_embedded_chunk, k)

            # Aggregate trustworthiness values for each k
            T_k_aggregate += T_k_chunk

            del X_chunk
            del X_embedded_chunk

        
        # Normalize the aggregated trustworthiness values
        T_k_aggregate = 1.0 - T_k_aggregate * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )

        # Append the result for the current map to the list
        trustworthiness_list.append(T_k_aggregate)

        del df_tsne_map
        del X_embedded

    return trustworthiness_list
