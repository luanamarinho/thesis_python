from sklearn.metrics import pairwise_distances_chunked
#from utils.compute_metrics_1k import trustworthiness_chunks
#from utils.trustworthiness_chunks_1k import trustworthiness_chunks
#from utils.trustworthiness_chunks_multik import trustworthiness_chunks_multik
import numpy as np
from joblib import load, dump
import os
import sys
from time import time

# Example dataset with 39970 rows and 2 columns
#folder =  "/home/luana/Documents/thesis_data"
folder = "C:/Users/luana/Documents/data"
#data = np.random.rand(39970, 2)
#dump(data, os.path.join(folder,"df_tsne.joblib"))
data_tsne = load(os.path.join(folder,"df_tsne.joblib"), mmap_mode='r')
data = data_tsne[:,:2]
sys.getsizeof(data) #160

# Define a function to process each chunk
def process_chunk(chunk, start):
    # Here, you can process each chunk, for example:
    # - Collect the chunks to build the full distance matrix
    # - Save the chunks to disk
    # - Perform some analysis on the chunk
    print(f"Processing chunk starting at row {start}")
    # For demonstration purposes, we'll just return the chunk
    return chunk

# Compute the pairwise distances in chunks
#working_memory Parameter: Specifies the size of the chunks (in MiB) to use during the computation. Adjust this based on your system's memory capacity.
# sklearn.get_config()['working_memory'] = 1024 by default
result_chunks = list(pairwise_distances_chunked(data, reduce_func=process_chunk, n_jobs=-1)) #working_memory=64
sys.getsizeof(result_chunks) #184
len(result_chunks) #12
result_chunks_tsne = result_chunks
del result_chunks

D_chunk_tsne = next(pairwise_distances_chunked(data, reduce_func=process_chunk, n_jobs=-1, working_memory=64))
D_chunk_tsne.shape #(209, 39970) with memory 64

# To collect all chunks into a full distance matrix (if needed)
start = time()
distance_matrix = np.vstack(result_chunks)
time() - start # 104
sys.getsizeof(distance_matrix) #12780807328 size of full loaded distance matrix
del distance_matrix


#dist_input_filepath = "/home/luana/Documents/thesis_data/dist_X.joblib"
dist_input_filepath = "C:/Users/luana/Documents/data/dist_X.joblib"

data_X = load(dist_input_filepath, mmap_mode='r')

def process_chunk2(chunk, start, data_input, k1, k2):
    print(f"Processing chunk starting at row {start}")
    #print(f"Length of chunk {len(chunk)}")
    data_input_chunk = data_input[start:start+len(chunk),:].copy()
    t1 = trustworthiness_chunks(dist_X=data_input_chunk, dist_embedded=chunk, k = k1)
    t2 = trustworthiness_chunks(dist_X=data_input_chunk, dist_embedded=chunk, k = k2)
    #print(f"Length of data input chunk {len(data_input_chunk)}")
    print(f"t1: {t1}, t2: {t2}")
    
    #return t1, t2
    return np.full(len(chunk), t1), np.full(len(chunk), t2)

## Use a lambda function to pass the additional argument
reduce_func = lambda chunk, start: process_chunk2(chunk, start, data_X, 30, 300)
#output = list(pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64)) 
#trustworthiness_scores = []
t1_sum,t2_sum = 0,0

for chunk in pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64):
    #trustworthiness_scores.append(np.unique(chunk)[0])  # Append the unique value of t
    t1_sum += np.unique(chunk[0])[0]
    t2_sum += np.unique(chunk[1])[0]

#>>> t_sum (t1_sum)
#23940419900
#>>> sum(trustworthiness_scores)
#23940419900
#>>> t1_sum
#23940419900
#>>> t2_sum
#236070634586


def process_chunk3(chunk, start, data_input, k):
  print(f"Processing chunk starting at row {start}")
  #print(f"Length of chunk {len(chunk)}")
  data_input_chunk = data_input[start:start+len(chunk),:].copy()
  #t1 = trustworthiness_chunks(dist_X=data_input_chunk, dist_embedded=chunk, k = k1)
  #t2 = trustworthiness_chunks(dist_X=data_input_chunk, dist_embedded=chunk, k = k2)
  t = trustworthiness_chunks_multik(dist_X=data_input_chunk, dist_embedded=chunk, k = k)
  #print(f"Length of data input chunk {len(data_input_chunk)}")
  print(f"t: {t}")
  t_chunk = tuple(np.full(len(chunk), t_value) for t_value in t)
  #return t1, t2
  return t_chunk

k = [30, 300]
## Use a lambda function to pass the additional argument
reduce_func = lambda chunk, start: process_chunk3(chunk, start, data_X, k)
#output = list(pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64)) 
#trustworthiness_scores = []
t_sum_k = np.zeros(len(k))

start = time()
for chunk in pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64):
  for i in range(len(k)):
      t_sum_k[i] += np.unique(chunk[i])[0]
runtime = time() - start


from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import numpy as np

# Generate random data
data_X1 = np.random.RandomState(42).rand(800, 50)
dist_X = pairwise_distances(data_X1)
dist_X.shape
k = [30, 300]

# Create a t-SNE map
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data_X1)
tsne_results.shape
dist_embedded = pairwise_distances(tsne_results)
dist_embedded.shape
from utils.trustworthiness_chunks_multik import trustworthiness_chunks_multik
output = trustworthiness_chunks_multik(dist_X=dist_X, dist_embedded=dist_embedded, k=[30, 300]) 
output


reduce_func_test = lambda chunk, start: process_chunk3(chunk, start, dist_X, k)
#output = list(pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64)) 
#trustworthiness_scores = []
t_sum_k = np.zeros(len(k))
n_samples = dist_embedded.shape[0]
trust = np.zeros(len(k))

for chunk in pairwise_distances_chunked(tsne_results, reduce_func=reduce_func_test, n_jobs=-1, working_memory=64):
  for i in range(len(k)):
      t_sum_k[i] += np.unique(chunk[i])[0]
      if k[i] >= n_samples // 2:
         raise ValueError(f"Nbr neighbors ({k[i]}) should be less than n_samples / 2 ({n_samples // 2})")
      trust[i] = 1.0 - t_sum_k[i] * (2.0 / (n_samples * k[i] * (2.0 * n_samples - 3.0 * k[i] - 1.0)))

print(trust)
from sklearn.manifold import trustworthiness
trustworthiness(X=data_X1, X_embedded=tsne_results, n_neighbors=30)
trustworthiness(X=data_X1, X_embedded=tsne_results, n_neighbors=300)

from utils.compute_trustworthiness_chunked import compute_trustworthines
compute_trustworthines(dist_X=dist_X, data_tsne=tsne_results, k=[30,300])

data_X.shape, data.shape
compute_trustworthines(dist_X=data_X, data_tsne=data, k=[30,300])
#array([0.63435042, 0.61001099])

for i in range(0, df_tsne.shape[1], 2):
    tsne_map = df_tsne.iloc[:, i:i+2].values  # Extract the TSNE map (2 columns)
    result = compute_trustworthiness(dist_X, tsne_map, k_values)
    results.append(result)
    print(f"TSNE map columns {i} and {i+1} processed: {result}"