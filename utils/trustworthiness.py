from sklearn.metrics import pairwise_distances
import numpy as np

def trustworthiness_ratio(dist_X, dist_embedded, min_k = 1, max_k = 70, metric = 'precomputed'):
    """
    Parameters:
    X: if metric = 'precomputed', a [n_samples, n_samples ]matrix-like object, ie, the distance matrix of the original data set.
    X_embedded: the embedded lower-dimensional space
    min_k: the minimum value of K in range of the k-NN to be visited
    max_k: the maximum value of the in the same range.

    Examples
    --------
    >>> from sklearn.manifold import TSNE
    >>> from sklearn.metrics import pairwise_distances
    >>> from sklearn.manifold import trustworthiness
    >>> X = np.random.RandomState(42).random((100, 50))  # Example high-dimensional data
    >>> X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    >>> dist_X = pairwise_distances(X, metric='euclidean')
    >>> dist_embedded = pairwise_distances(X_embedded, metric='euclidean')
    >>> print(trustworthiness_ratio(dist_X, dist_embedded, min_k=1, max_k=10, metric='precomputed'))
    ([0.9638775510204082, 0.9106217616580311, 0.891859649122807, 0.8721122994652406, 0.8451086956521738, 0.8327071823204419, 0.8154253611556983, 0.8084285714285715, 0.7922093023255814, 0.7808994082840237], 0.839)
    >>> trustworthiness(X, X_embedded, n_neighbors=5)
    0.8451086956521738
    """
    T_k = []
    C_k = []
    n_samples = dist_X.shape[0]
    if max_k >= n_samples / 2:
      raise ValueError(
        f"k ({max_k}) should be less than n_samples / 2"
        f" ({n_samples / 2})"
      )
    
    dist_X = pairwise_distances(dist_X, metric=metric)
    if metric == "precomputed":
        dist_X = dist_X.copy()
    # Diagonal is set to np.inf to exclude the points themselves from their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    # `ind_X[i]` is the index of sorted distances between i and other samples
   
    dist_embedded = pairwise_distances(dist_embedded, metric=metric)
    if metric == "precomputed":
        dist_embedded = dist_embedded.copy()
    ind_X_embedded = np.argsort(dist_embedded, axis=1)[:,1:]

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]

    # Loop over k
    for k in range(min_k, max_k+1):
      ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded[:,:k]] - k
      )
      t = np.sum(ranks[ranks > 0])
      t = 1.0 - t * (
        2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
      )
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
    
    median_first_25_percent = np.median(T_k[:int(0.25*max_k)])
    start_index = max_k - int(0.25*max_k)
    median_last_25_percent = np.median(T_k[start_index:])
    trustworthiness_ratio = median_last_25_percent/median_first_25_percent

    median_first_25_percent_C = np.median(C_k[:int(0.25 * max_k)])
    start_index_C = max_k - int(0.25 * max_k)
    median_last_25_percent_C = np.median(C_k[start_index_C:])
    continuity_ratio = median_last_25_percent_C / median_first_25_percent_C
    return T_k, trustworthiness_ratio, C_k, continuity_ratio


