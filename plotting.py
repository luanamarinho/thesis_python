from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(1234)
ind_to_sample = np.random.choice(39970, size=20000, replace=False)

output= load('output/pipeline_multiples_test2.joblib')
output = load('output/pipeline_multiples_perp25_0-45.joblib')
output = load('output/perp25/pipeline_multiples_perp25_137-180.joblib')
metadata = pd.read_csv('data/metadata_sampled.csv')

len(output)
len(output[0])
affinity_runtime_sec = output[1] #337
for elem in output[0]:
    print(elem[0])

i = 10
combination =  output[0][i][0]
runtime_sec = output[0][i][2]
tsne_df = output[0][i][1]

KL_divergence = output[0][i][3]

runtime_all = [tuple[2] for tuple in output[0]]
combinations_all = [tuple[0] for tuple in output[0]]
mean_runtime_min = np.mean(runtime_all)/60

embedding_X, embedding_Y = tsne_df.iloc[:, 0], tsne_df.iloc[:, 1]
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
title = f"TSNE Plot for combination {combination}"
plt.title(title)
plt.show()

