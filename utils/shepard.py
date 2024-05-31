import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_chunked


def shepard_diagram(original_distances, tsne_configuration, n_jobs = 2, X = None):
  """
  Calculate pairwise distances and plot the Shepard diagram
  Parameters:
  original_distances: Pre-computed pairwise distances in original high dimensional dataset
  X: High-dimensional data. If provided, original distances will be computed from this data.
  tsne_configuration: t-SNE configuration of the data.
  """
  if X is not None:
    original_distances = next(pairwise_distances_chunked(X, n_jobs = n_jobs))
  elif original_distances is None:
    raise ValueError("Either original_distances or X must be provided.")
  
  if isinstance(tsne_configuration, pd.DataFrame):
    tsne_configuration = tsne_configuration.values
  tsne_distances = next(pairwise_distances_chunked(tsne_configuration, n_jobs = n_jobs))

  original_distances_flat = original_distances[np.triu_indices_from(original_distances, 1)]
  tsne_distances_flat = tsne_distances[np.triu_indices_from(tsne_distances, 1)]

  plt.figure(figsize=(8, 6))
  plt.plot([0, max(original_distances_flat)], [0, max(original_distances_flat)], 'r--', label='Identity line')
  plt.scatter(original_distances_flat, tsne_distances_flat, alpha=0.5)
  plt.xlabel('Original Distances')
  plt.ylabel('t-SNE Distances')
  plt.title('Shepard Diagram')
  plt.show()
