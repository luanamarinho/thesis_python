from sklearn.metrics import pairwise_distances
import numpy as np

def trustworthiness_continuity(dist_X, dist_embedded, min_k=1, max_k=70, metric='precomputed'):
    T_k = []
    C_k = []
    n_samples = dist_X.shape[0]
    if max_k >= n_samples / 2:
        raise ValueError(
            f"k ({max_k}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )

    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = np.argsort(dist_embedded, axis=1)[:, 1:]

    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples)
    ind_X_reshaped = ind_X.reshape(-1)  # Flatten the index array
    inverted_index[ordered_indices, ind_X_reshaped] = np.tile(np.arange(1, n_samples + 1), n_samples)

    for k in range(min_k, max_k + 1):
        # Adjust the indices for the embedded space to match chunking
        ind_X_embedded_reshaped = ind_X_embedded[:, :k].reshape(-1)
        ranks = (inverted_index[ordered_indices[:, np.newaxis], ind_X_embedded_reshaped] - k)
        ranks = ranks.reshape(n_samples, k)
        
        t = np.sum(ranks[ranks > 0])
        t = 1.0 - t * (2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0)))
        T_k.append(t)

        continuity_score = 0
        for i in range(n_samples):
            set_high_dim = set(ind_X[i, :k])
            set_low_dim = set(ind_X_embedded[i, :k])
            U_i_k = set_low_dim - set_high_dim
            if len(U_i_k) > 0:
                ranks = np.array([np.where(ind_X[i] == j)[0][0] for j in U_i_k])
                continuity_score += np.sum(ranks - k)
        normalization_factor = 2 / (n_samples * k * (2 * n_samples - 3 * k - 1))
        continuity_score = 1 - normalization_factor * continuity_score
        C_k.append(continuity_score)

    median_first_25_percent = np.median(T_k[:int(0.25 * max_k)])
    start_index = max_k - int(0.25 * max_k)
    median_last_25_percent = np.median(T_k[start_index:])
    trustworthiness_ratio = median_last_25_percent / median_first_25_percent

    median_first_25_percent_C = np.median(C_k[:int(0.25 * max_k)])
    start_index_C = max_k - int(0.25 * max_k)
    median_last_25_percent_C = np.median(C_k[start_index_C:])
    continuity_ratio = median_last_25_percent_C / median_first_25_percent_C

    return T_k, trustworthiness_ratio, C_k, continuity_ratio
