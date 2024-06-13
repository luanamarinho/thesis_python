from utils.trustworthiness_chunks_multik import trustworthiness_chunks_multik
from sklearn.metrics import pairwise_distances_chunked
import numpy as np

def compute_trustworthiness(dist_X, data_tsne, k, n_jobs=-1, working_memory=64):
  t_sum_k, trust = np.zeros(len(k)), np.zeros(len(k))
  n_samples = data_tsne.shape[0]
  reduce_func = lambda chunk, start: process_chunk(chunk, start, dist_X, k)

  for t_chunk in pairwise_distances_chunked(data_tsne, reduce_func=reduce_func, n_jobs=n_jobs, working_memory=working_memory):
    for i in range(len(k)):
      if k[i] >= n_samples // 2:
        raise ValueError(f"Nbr neighbors ({k[i]}) should be less than n_samples / 2 ({n_samples // 2})")
      
      t_sum_k[i] += np.unique(t_chunk[i])[0]    
      trust[i] = 1.0 - t_sum_k[i] * (2.0 / (n_samples * k[i] * (2.0 * n_samples - 3.0 * k[i] - 1.0)))

  return trust

def process_chunk(chunk, start, data_input, k):
  print(f"Processing chunk starting at row {start}")
  data_input_chunk = data_input[start:start+len(chunk),:].copy()
  t = trustworthiness_chunks_multik(dist_X=data_input_chunk, dist_embedded=chunk, k = k)
  t_chunk = tuple(np.full(len(chunk), t_value) for t_value in t)

  return t_chunk