import numpy as np

def trustworthiness_continuity(dist_X_chunk, dist_embedded_chunk, min_k=1, max_k=70, metric='precomputed'):
    T_k_chunk = []
    n_samples_chunk = dist_X_chunk.shape[0]

    # Your existing implementation for trustworthiness computation
    for k in range(min_k, max_k + 1):
        # Compute trustworthiness for the chunk
        inverted_index = np.zeros((n_samples_chunk, n_samples_chunk), dtype=int)
        ordered_indices = np.arange(n_samples_chunk + 1)
        inverted_index[ordered_indices[:-1, np.newaxis], np.argsort(dist_X_chunk, axis=1)] = ordered_indices[1:]

        ranks = (
            inverted_index[ordered_indices[:-1, np.newaxis], np.argsort(dist_embedded_chunk, axis=1)[:, :k]] - k
        )
        t = np.sum(ranks[ranks > 0])
        T_k_chunk.append(t)  # Append unnormalized trustworthiness value

    return T_k_chunk
 