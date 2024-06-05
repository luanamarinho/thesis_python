import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from joblib import dump
import os


n_jobs=1
df_tsne = pd.read_csv('output/df_tsne_unique.csv', compression='gzip')
k = 0
dist_tsne = []
for i in np.arange(int(df_tsne.shape[1]/2)):
    #print(f'[:,{k}:{k+2}]')
    tsne_configuration = df_tsne.iloc[:,k:k+2]
    if isinstance(tsne_configuration, pd.DataFrame):
        tsne_configuration = tsne_configuration.values
    
    # Initialize the distance matrix
    n_samples = tsne_configuration.shape[0]
    full_distance_matrix = np.zeros((n_samples, n_samples))

    for start_idx, chunk in enumerate(pairwise_distances_chunked(tsne_configuration, n_jobs=n_jobs)):
        end_idx = start_idx + chunk.shape[0]
        full_distance_matrix[start_idx:end_idx] = chunk
    
    tsne_dist_name = f'tsne_columns[:,{k}:{k+2}].joblib'
    file_path = os.path.join('output','dist', tsne_dist_name)
    dump(full_distance_matrix, file_path)
    
    k = k+2



