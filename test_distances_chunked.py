from sklearn.metrics import pairwise_distances_chunked
from utils.compute_metrics_1k import trustworthiness_chunks
import numpy as np
from joblib import load
import os
import sys
from time import time

# Example dataset with 40,000 rows and 2 columns
folder =  "C:/Users/luana/Documents/data"
#data = np.random.rand(40000, 2)
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


dist_input_filepath = "C:/Users/luana/Documents/data/dist_X.joblib"
data_X = load(dist_input_filepath, mmap_mode='r')

def process_chunk2(chunk, start, data_input, k):
    print(f"Processing chunk starting at row {start}")
    #print(f"Length of chunk {len(chunk)}")
    data_input_chunk = data_input[start:start+len(chunk),:].copy()
    t = trustworthiness_chunks(dist_X=data_input_chunk, dist_embedded=chunk, k = k)
    #print(f"Length of data input chunk {len(data_input_chunk)}")
    print(t)
    return np.full(len(chunk), t)

## Use a lambda function to pass the additional argument
reduce_func = lambda chunk, start: process_chunk2(chunk, start, data_X, 30)
#output = list(pairwise_distances_chunked(data, reduce_func=reduce_func, n_jobs=-1, working_memory=64)) 
trustworthiness_scores = []

for chunk in pairwise_distances_chunked(data_X, reduce_func=reduce_func, n_jobs=-1, working_memory=64):
    trustworthiness_scores.append(np.unique(chunk)[0])  # Append the unique value of t