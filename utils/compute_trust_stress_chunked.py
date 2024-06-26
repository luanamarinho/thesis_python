from utils.trustworthiness_chunks_multik import trustworthiness_chunks_multik
from utils.shepard_stress_chunks import shepard_stress_chunk
from sklearn.metrics import pairwise_distances_chunked
import numpy as np

def compute_stress_trust(dist_X, data_tsne, k, n_jobs=-1, working_memory=64):
  t_sum_k, trust = np.zeros(len(k)), np.zeros(len(k))
  sum_num_denom = np.zeros(2)
  n_samples = data_tsne.shape[0]
  reduce_func = lambda chunk, start: process_chunk(chunk, start, dist_X, k)

  for chunk in pairwise_distances_chunked(data_tsne, reduce_func=reduce_func, n_jobs=n_jobs, working_memory=working_memory):
    #print(len(chunk)
    chunk_stress = chunk[:2] 
    chunk_t = chunk[2:4]
    for i in range(len(k)):
      if k[i] >= n_samples // 2:
        raise ValueError(f"Nbr neighbors ({k[i]}) should be less than n_samples / 2 ({n_samples // 2})")
      
      t_sum_k[i] += chunk_t[i]    
      trust[i] = 1.0 - t_sum_k[i] * (2.0 / (n_samples * k[i] * (2.0 * n_samples - 3.0 * k[i] - 1.0)))
   
  for j in range(len(sum_num_denom)):
    sum_num_denom[j] += chunk_stress[j]

  stress = np.sqrt(sum_num_denom[0] / sum_num_denom[1])

  return trust, stress, sum_num_denom[0]

def process_chunk(chunk, start, data_input, k):
  print(f"Processing chunk starting at row {start}")
  data_input_chunk = data_input[start:start+len(chunk),:].copy()
  t = trustworthiness_chunks_multik(dist_X=data_input_chunk, dist_embedded=chunk, k = k)
  stress = shepard_stress_chunk(dist_X=data_input_chunk, dist_embedded=chunk)
  metrics = stress + t
  metrics_chunk = np.zeros(len(chunk))
    
  for i in range(len(metrics)):
        metrics_chunk[i] = metrics[i]
  
  return metrics_chunk