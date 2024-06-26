from utils.shepard_stress_chunks import shepard_stress_chunk
from sklearn.metrics import pairwise_distances_chunked
import numpy as np

def compute_shepard_stress(dist_X, data_tsne, n_jobs=-1, working_memory=64):
  """
    Compute the shepard stress of an embedding.

    Parameters:
    - dist_X: distance matrix of the input data
    - data_tsne: tsne map
    - n_jobs: number of threads
    - working memory: argument controls the size of the chunk

    Returns:
    - Shepard stress (float) and the numerator of the ratio for comparision
    """
  sum_num_denom = np.zeros(2)
  reduce_func = lambda chunk, start: process_chunk(chunk, start, dist_X)

  for chunk in pairwise_distances_chunked(data_tsne, reduce_func=reduce_func, n_jobs=n_jobs, working_memory=working_memory):    
    for j in range(len(sum_num_denom)):
        sum_num_denom[j] += np.unique(chunk[j])[0]

  stress = np.sqrt(sum_num_denom[0] / sum_num_denom[1]) 

  return [stress, sum_num_denom[0]]

def process_chunk(chunk, start, data_input):
  print(f"Processing chunk starting at row {start}")
  data_input_chunk = data_input[start:start+len(chunk),:].copy()
  stress_chunk = shepard_stress_chunk(dist_X=data_input_chunk, dist_embedded=chunk)
  stress_chunk_full = tuple(np.full(len(chunk), stress_elem) for stress_elem in stress_chunk)

  return stress_chunk_full