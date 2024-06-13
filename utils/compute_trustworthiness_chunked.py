from utils.trustworthiness_chunks_multik import trustworthiness_chunks_multik
import numpy as np

def compute_trustworthines(dist_X, data):
   
reduce_func_test = lambda chunk, start: process_chunk3(chunk, start, dist_X, k)
#output = list(pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64)) 
#trustworthiness_scores = []
t_sum_k = np.zeros(len(k))
n_samples = dist_embedded.shape[0]
trust = np.zeros(len(k))

for chunk in pairwise_distances_chunked(tsne_results, reduce_func=reduce_func_test, n_jobs=-1, working_memory=64):
  for i in range(len(k)):
      t_sum_k[i] += np.unique(chunk[i])[0]
      if k[i] >= n_samples // 2:
         raise ValueError(f"Nbr neighbors ({k[i]}) should be less than n_samples / 2 ({n_samples // 2})")
      trust[i] = 1.0 - t_sum_k[i] * (2.0 / (n_samples * k[i] * (2.0 * n_samples - 3.0 * k[i] - 1.0)))

def process_chunk3(chunk, start, data_input, k):
  print(f"Processing chunk starting at row {start}")
  data_input_chunk = data_input[start:start+len(chunk),:].copy()
  t = trustworthiness_chunks_multik(dist_X=data_input_chunk, dist_embedded=chunk, k = k)
  t_chunk = tuple(np.full(len(chunk), t_value) for t_value in t)

  return t_chunk