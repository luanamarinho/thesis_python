from time import time
from utils.compute_metrics import compute_metrics


# Example usage
#compute_metrics(df_tsne_filepath = 'output/df_tsne_unique.csv', dist_X_filepat = 'output/original_data.csv', max_k=100)

start = time()
output_all = compute_metrics(df_tsne_filepath="output/df_tsne_unique.csv", dist_X_filepath="C:\\Users\\luana\\Documents\\data\\dist_X.joblib", max_k=50)
runtime = time() - start


# Usage example
output_all = compute_metrics(
    df_tsne_filepath="output/df_tsne_unique.csv",
    dist_X_filepath="C:\\Users\\luana\\Documents\\data\\dist_X.joblib",
    min_k=1,
    max_k=100,
    chunk_size=1000
)
print(output_all)






import numpy as np
import pandas as pd
from joblib import load
import gzip
from time import time
from sklearn.metrics import pairwise_distances
#from sklearn.manifold import trustworthiness
from utils.trustworthiness_chunks import trustworthiness
from utils.compute_distances import compute_distance_matrix_chunked


def compute_metrics(dist_input_filepath, df_tsne_filepath, chunk_size=1000, min_k=1, max_k=50):
  # Load input distance matrix
  X = load(dist_input_filepath)
  n_samples = X.shape[0]

  # Load t-SNE maps and compute distances
  df_tsne = pd.read_csv(df_tsne_filepath, compression='gzip')
  df_tsne = df_tsne.iloc[:, 0:2] # testing first tsne map
  X_embedded = compute_distance_matrix_chunked(df_tsne)

  # Initialize variables to store trustworthiness results
  T_k_aggregate = np.zeros(max_k - min_k + 1)

  # Split data into chunks
  chunks = [(X[i:i+chunk_size], X_embedded[i:i+chunk_size]) for i in range(0, len(df_tsne), chunk_size)]


  # Compute trustworthiness for each chunk
  for chunk in chunks:
    # Compute trustworthiness for the chunk for each k
    T_k_chunk = trustworthiness(chunk[0], chunk[1], min_k=min_k, max_k=max_k)

    # Aggregate trustworthiness values for each k
    T_k_aggregate += T_k_chunk

  # Normalize the aggregated trustworthiness values
  for k in range(min_k, max_k + 1):
    T_k_aggregate[k - min_k] = 1.0 - T_k_aggregate[k - min_k] * (
      2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
    )

    return T_k_aggregate


df_tsne_filepath="output/df_tsne_unique.csv"
dist_input_filepath="C:\\Users\\luana\\Documents\\data\\dist_X.joblib"
from utils.compute_metrics_1k import compute_metrics

start = time()
output = compute_metrics(dist_input_filepath=dist_input_filepath, df_tsne_filepath=df_tsne_filepath, max_k=2)
runtime = time() - start
output

dist_input_filepath = "dist_X_toy.joblib"
df_tsne_filepath = "X_embedded.csv"

output = compute_metrics(dist_input_filepath=dist_input_filepath, df_tsne_filepath=df_tsne_filepath, max_k=2, chunk_size=50)
output
from sklearn.manifold import trustworthiness
test = trustworthiness()



