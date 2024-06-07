import numpy as np
import pandas as pd
from joblib import load
from utils.trustworthiness_chunks_1k import trustworthiness_chunks
from utils.compute_distances import compute_distance_matrix_chunked

def compute_metrics(dist_input_filepath, df_tsne_filepath, chunk_size=1000, k=1):
  # Load input distance matrix
  X = load(dist_input_filepath)
  n_samples = X.shape[0]

  # Load t-SNE maps and compute distances
  df_tsne = pd.read_csv(df_tsne_filepath, compression='gzip')
  df_tsne = df_tsne.iloc[:, 0:2] # testing first tsne map
  X_embedded = compute_distance_matrix_chunked(df_tsne)

  # Initialize variables to store trustworthiness results
  T_k_aggregate = np.zeros(1)

  # Split data into chunks
  chunks = [(X[i:i+chunk_size], X_embedded[i:i+chunk_size]) for i in range(0, len(df_tsne), chunk_size)]


  # Compute trustworthiness for each chunk
  for chunk in chunks:
    # Compute trustworthiness for the chunk for each k
    T_k_chunk = trustworthiness_chunks(chunk[0], chunk[1], k)

    # Aggregate trustworthiness values for each k
    T_k_aggregate += T_k_chunk

  # Normalize the aggregated trustworthiness values
  T_k_aggregate = 1.0 - T_k_aggregate * (
      2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
  )

  return T_k_aggregate
