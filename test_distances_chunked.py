from sklearn.metrics import pairwise_distances_chunked
#from utils.compute_metrics_1k import trustworthiness_chunks
#from utils.trustworthiness_chunks_1k import trustworthiness_chunks
#from utils.trustworthiness_chunks_multik import trustworthiness_chunks_multik
import numpy as np
from joblib import load, dump
import os
import sys
from time import time
import gzip

# Example dataset with 39970 rows and 2 columns
#folder =  "/home/luana/Documents/thesis_data"
folder = "C:/Users/luana/Documents/data"
#data = np.random.rand(39970, 2)
#dump(data, os.path.join(folder,"df_tsne.joblib"))

# Dist input data
data_file_path = 'C:/Users/luana/Documents/data/data_preprocessed_5000_10HVG'#'/home/luana/workspace/data/data_preprocessed_40000_10HVG'
#with gzip.GzipFile(data_file_path, "r") as data_file:
#  data = np.load(data_file)
data = load('output/df_tsne_momentum.joblib.gz')
#data_tsne = load(os.path.join(folder,"df_tsne.joblib"), mmap_mode='r')
#data = data_tsne[:,:2]
sys.getsizeof(data) #39402164 for tsne maps and 134086328 for input data

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
sys.getsizeof(result_chunks) #88
len(result_chunks) #1
#result_chunks_tsne = result_chunks
#del result_chunks

#D_chunk_tsne = next(pairwise_distances_chunked(data, reduce_func=process_chunk, n_jobs=-1, working_memory=64))
#D_chunk_tsne.shape #(209, 39970) with memory 64

# To collect all chunks into a full distance matrix (if needed)
start = time()
distance_matrix = np.vstack(result_chunks)
time() - start # 104
sys.getsizeof(distance_matrix) #198005128 size of full loaded distance matrix of X
#del distance_matrix
#dump(distance_matrix, "C:/Users/luana/Documents/data/dist_X_5000.joblib")


#dist_input_filepath = "/home/luana/Documents/thesis_data/dist_X.joblib"
dist_input_filepath = "C:/Users/luana/Documents/data/dist_X_5000.joblib"

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

from utils.compute_trustworthiness_chunked import compute_trustworthiness
compute_trustworthiness(dist_X=dist_X, data_tsne=tsne_results, k=[30,300]) #array([0.66272311, 0.61871706])

data_X.shape, data.shape
compute_trustworthiness(dist_X=data_X, data_tsne=data, k=[30,300])
#array([0.63435042, 0.61001099])




for i in range(0, df_tsne.shape[1], 2):
    tsne_map = df_tsne.iloc[:, i:i+2].values  # Extract the TSNE map (2 columns)
    result = compute_trustworthiness(dist_X, tsne_map, k_values)
    results.append(result)
    print(f"TSNE map columns {i} and {i+1} processed: {result}")
          

def shepard_stress_chunk(dist_X, dist_embedded):
  """
  Compute Shepard stress for a chunk of the distance matrices.
    
  Parameters:
  original_distances (numpy.ndarray): A chunk of pairwise distances in the original high-dimensional space.
  reduced_distances (numpy.ndarray): A chunk of pairwise distances in the reduced low-dimensional space.
  metric (bool): Whether to use the metric version of MDS.
    
  Returns:
  tuple: Shepard stress numerator and denominator for the chunk.
    """

  #sim_flat = dist_X.ravel() # from scikit learn
  #dis_flat = dist_embedded.ravel()
  dist_X_flat = dist_X[np.triu_indices_from(dist_X, 1)] # Considering only the upper triangle
  dist_embeddeds_flat = dist_embedded[np.triu_indices_from(dist_embedded, 1)]

  stress_numerator = ((dist_embeddeds_flat - dist_X_flat) ** 2).sum()
  stress_denominator = (dist_X_flat ** 2).sum()
    
  return stress_numerator, stress_denominator

shepard_stress_chunk(dist_X, dist_embedded) #(141261635.69419235, 2654164.2243823246)

def process_chunk_shepard(chunk, start, data_input):
  print(f"Processing chunk starting at row {start}")
  data_input_chunk = data_input[start:start+len(chunk),:]
  stress_chunk = shepard_stress_chunk(dist_X=data_input_chunk, dist_embedded=chunk)
  stress_chunk_full = tuple(np.full(len(chunk), stress_value) for stress_value in stress_chunk)

  return stress_chunk_full

reduce_func = lambda chunk, start: process_chunk_shepard(chunk, start, dist_X)
sum_num_denom = np.zeros(2)

start = time()
for chunk in pairwise_distances_chunked(tsne_results, reduce_func=reduce_func, n_jobs=-1, working_memory=64):
  for i in range(len(sum_num_denom)):
    sum_num_denom[i] += np.unique(chunk[i])[0]
runtime = time() - start

stress = np.sqrt(sum_num_denom[0] / sum_num_denom[1])

from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
# Fit the data and perform dimensionality reduction
mds_transformed = mds.fit_transform(dist_X)
mds_transformed.shape

sum_num_denom = np.zeros(2)

start = time()
for chunk in pairwise_distances_chunked(mds_transformed, reduce_func=reduce_func, n_jobs=-1, working_memory=64):
  for i in range(len(sum_num_denom)):
    sum_num_denom[i] += np.unique(chunk[i])[0]
runtime = time() - start

stress = np.sqrt(sum_num_denom[0] / sum_num_denom[1]) # 0.40459501846416124
mds.stress_
sum_num_denom[0] / 2

from utils.compute_trust_stress_chunked import compute_metrics
compute_metrics(dist_X, mds_transformed, [3])

from utils.compute_trustworthiness_chunked import compute_trustworthiness
compute_trustworthiness(dist_X, mds_transformed, [3])
compute_trustworthiness(dist_X, tsne_results, [3])
compute_trustworthiness(dist_X, tsne_map, [3])
compute_trustworthiness(dist_X, tsne_map, k_values)



################
#1.1) test compute_trustworthiness with random data + MDS. Compare with skicit learn function
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances_chunked
from utils.compute_trustworthiness_chunked import compute_trustworthiness
from utils.compute_shepard_stress_chunked import compute_shepard_stress
from utils.compute_trust_stress_chunked import compute_stress_trust


data_X1 = np.random.RandomState(42).rand(800, 50)
dist_X = pairwise_distances(data_X1)
dist_X.shape

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_transformed = mds.fit_transform(dist_X)
mds_transformed.shape

# scikit learn
print(trustworthiness(X=data_X1, X_embedded=mds_transformed, n_neighbors=30),
      trustworthiness(X=data_X1, X_embedded=mds_transformed, n_neighbors=300)
)
# 0.5864097084161697 0.6565014902241297

# compute_trustworthiness
print(
   compute_trustworthiness(dist_X=dist_X,
                           data_tsne=mds_transformed,
                           k=[30,300])
)
# [0.58640971 0.65650149]



#2.1) test compute_shepard with random data + MDS. Compare with skicit learn function 
# scikit learn
print(mds.stress_) #434527.12288466265 this is the raw, unnormalized (d - dmap)^2 (our numerator)

# compute_shepard_stress: returns stress ratio and its numerator
compute_shepard_stress(dist_X=dist_X, data_tsne=mds_transformed) #[0.40459501846416124, 434479.06333569647]


# 3.1) test compute_shepard_trust
#compute_stress_trust(dist_X=dist_X,
                     data_tsne=mds_transformed,
                     k=[30,300]
) # not right


from utils.compute_trust_stress_chunked import compute_stress_trust
compute_stress_trust(dist_X=dist_X, data_tsne=mds_transformed,k=[30,300]) #(array([0.58640971, 0.65650149]), 0.40459501846416124, 434479.06333569647)
trust, stress,_ = compute_stress_trust(dist_X=dist_X, data_tsne=mds_transformed,k=[30,300])
result = np.concatenate([trust, [stress]])

from utils.compute_trust_stress_chunked_norm import compute_stress_trust
compute_stress_trust(dist_X=dist_X, data_tsne=mds_transformed,k=[30,300]) #(array([0.58640971, 0.65650149]), 0.40459501846416124, 434479.06333569647)


# Testing run_compute
dist_X.__class__
mds_transformed.__class__
from joblib import dump
dump(dist_X, 'output/dist_X_toy.joblib')
dump(mds_transformed, 'output/data_mds_toy.joblib')
run_compute_metrics(toy_data = True)
#(memmap([[0.58640971, 0.65650149]]), 'C:/Users/luana/Documents/data\\output_memmap_toy_0_2.dat', 1, 2)
run_compute_metrics(toy_data = True)
#(memmap([[0.58640971, 0.65650149]]), 'C:/Users/luana/Documents/data\\output_memmap_trust_toy_0_2.dat', memmap([0.40459502]), 'C:/Users/luana/Documents/data\\output_memmap_stress_toy_0_2.dat', 1, 2)

run_compute_metrics(end_index = 4)