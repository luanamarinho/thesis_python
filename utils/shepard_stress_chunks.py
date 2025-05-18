import numpy as np

def shepard_stress_chunk(dist_X, dist_embedded):
  """
  Compute Shepard stress for a chunk of the distance matrices.
    
  Parameters:
  original_distances (numpy.ndarray): A chunk of pairwise distances in the original high-dimensional space.
  reduced_distances (numpy.ndarray): A chunk of pairwise distances in the reduced low-dimensional space.
    
  Returns:
  tuple: Shepard stress numerator and denominator for the chunk.
  """

  #sim_flat = dist_X.ravel() # from scikit learn
  #dis_flat = dist_embedded.ravel()
  dist_X_flat = dist_X[np.triu_indices_from(dist_X, 1)] # Considering only the upper triangle
  dist_embedded_flat = dist_embedded[np.triu_indices_from(dist_embedded, 1)]

  stress_numerator = ((dist_embedded_flat - dist_X_flat) ** 2).sum()
  stress_denominator = (dist_X_flat ** 2).sum()
    
  return [stress_numerator, stress_denominator]
