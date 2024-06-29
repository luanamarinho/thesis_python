import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_tsne = pd.read_csv('output/df_tsne_unique.csv', compression='gzip')
metadata = pd.read_csv('inst/data/metadata_sampled.csv')

# for i in range(df_tsne.shape[1] // 2): print (1)
i = 0
df_tsne_map = df_tsne.iloc[:, 2*i:2*i+2]


embedding_X, embedding_Y = df_tsne_map.iloc[:, 0], df_tsne_map.iloc[:, 1]
combined_data = pd.DataFrame({
    'TSNE_1': embedding_X,
    'TSNE_2': embedding_Y,
    'CellType': metadata['CellType']
})
combined_data['CellType'] = pd.Categorical(combined_data['CellType'])


# Plot using DataFrame plot method
combined_data.plot.scatter(x='TSNE_1', y='TSNE_2', c='CellType', cmap = 'tab10')
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
title = f"TSNE Plot for combination"
plt.title(title)
plt.show()


high_k = 300
low_k = 30
see the combinations for the 93 KL
shepard stress
shepard plots with samples of the distance matrices
Laurens van der Maaten: point out mistake in error

cross table of the unique values


cross-table
many plots
dont show code..
material
explain parameters and metrics
explain 93
cluster with and without parameters
clustering hierarchical dbscan
endogeinty
contact after defense

    


