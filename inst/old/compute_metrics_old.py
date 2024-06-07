import numpy as np
import pandas as pd
from utils.compute_distances import compute_distance_matrix_chunked
from inst.old.trustworthiness import trustworthiness_continuity
from joblib import load


def compute_metrics(df_tsne_filepath, dist_X_filepath, min_k=1, max_k=50, n_jobs=1):

    # Load input pairwise distance matrix
    dist_X = load(dist_X_filepath)
    dist_X = dist_X.astype(np.float32)

    # Load t-SNE maps
    df_tsne = pd.read_csv(df_tsne_filepath, compression='gzip')
    
    j = 0
    output_metrics = []
    for i in np.arange(2): #int(df_tsne.shape[1]/2)
      #print(f'[:,{k}:{k+2}]')
      tsne_configuration = df_tsne.iloc[:,j:j+2]
      if isinstance(tsne_configuration, pd.DataFrame):
        tsne_configuration = tsne_configuration.values
      
      dist_embedded = compute_distance_matrix_chunked(data=tsne_configuration)
      dist_embedded = dist_embedded.astype(np.float32)

      # Compute the trustworthiness and continuity
      output = trustworthiness_continuity(dist_X, dist_embedded, min_k, max_k, n_jobs)
      output_metrics.append(output)
      j = j + 2
      return(output_metrics)
