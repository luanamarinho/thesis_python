import numpy as np
from sklearn.metrics import pairwise_distances_chunked

def compute_distance_matrix_chunked(data, n_jobs=1):
    n_samples = data.shape[0]
    full_distance_matrix = np.zeros((n_samples, n_samples))

    start_idx = 0
    for chunk in pairwise_distances_chunked(data, n_jobs=n_jobs):
        end_idx = start_idx + chunk.shape[0]
        full_distance_matrix[start_idx:end_idx] = chunk
        start_idx = end_idx

    return full_distance_matrix