import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming `X` is your dataset
# If your data isn't standardized, it's a good idea to scale it
df_metrics = pd.read_csv("output/df_metric_momentum_wresults.csv")
parameters = ['Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta']
outcomes = ['tSNE_runtime_min', 'KL_divergence', 'trust_k30', 'trust_k300', 'stress']
X = df_metrics[outcomes]
X_scaled = StandardScaler().fit_transform(X)


# Perform PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Define the range of k values you want to evaluate
k = 4  # For example, k from 2 to 9

# Plotting the k-distance graphs for each k
plt.figure(figsize=(12, 8))

nearest_neighbors = NearestNeighbors(n_neighbors=k)
neighbors = nearest_neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
    
# Sort the distances to plot the k-distance graph
distances = np.sort(distances[:, k-1], axis=0)
    
plt.plot(distances, label=f'k={k}')

plt.xlabel('Data points sorted by distance')
plt.ylabel('Distance to k-th nearest neighbor')
plt.title('k-distance graphs for different k values')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Set eps based on the k-distance graph (elbow point)
eps_value = 1.3  # Set based on the elbow in your k-distance graph

# Set min_samples based on the recommendation
min_samples_value = 11

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
dbscan.fit(X)

# Extract labels and analyze the result
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')

# Extract unique cluster labels
unique_labels = np.unique(labels)

# Define a colormap
n_labels = len(unique_labels) - 1 if -1 in unique_labels else len(unique_labels)  # Exclude noise for color map if present
colors = plt.get_cmap('tab10', n_labels)  # Adjust 'tab10' if more colors are needed

# Plot the clusters
plt.figure(figsize=(10, 6))
for label in unique_labels:
    if label == -1:
        # Noise points are usually colored in black
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                    color='k', label='Noise', edgecolor='k', s=50)
    else:
        # Normal clusters
        color = colors(label / n_labels)  # Normalize label to [0, 1] range
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                    color=color, label=f'Cluster {label}', edgecolor='k', s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering on PCA-Reduced Data')
plt.legend()
plt.grid(True)
plt.show()