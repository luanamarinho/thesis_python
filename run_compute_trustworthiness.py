import pandas as pd
import numpy as np
import time
from memory_profiler import memory_usage
from utils.compute_trustworthiness_chunked import compute_trustworthiness
from joblib import Parallel, delayed, load
import sys
import os


folder = "../data"
data_tsne = load(os.path.join(folder,"df_tsne.joblib"), mmap_mode='r')
data_X = load('os.path.join(folder,"dist_X.joblib")', mmap_mode='r')


# Values of k for k-nearest neighbors
k_values = [30, 300]

# Pre-allocate a memmap output array
num_tsne_maps = df_tsne.shape[1] // 2
output_filename_memmap = 'output_memmap.dat'
output = np.memmap(output_filename_memmap, dtype='float64', shape=(num_tsne_maps, len(k_values)), mode='w+')

# Function to process each TSNE map
def process_tsne_map(index, output):
    tsne_map = df_tsne.iloc[:, index:index+2].values  # Extract the TSNE map (2 columns)
    result = compute_trustworthiness(dist_X, tsne_map, k_values)
    output[index // 2] = result  # Write result to the pre-allocated memmap array

# Measure memory usage before the process
initial_memory = memory_usage(max_usage=True)

# Measure the start time
start_time = time.time()

# Parallel processing using joblib
Parallel(n_jobs=-1)(delayed(process_tsne_map)(i, output) for i in range(0, df_tsne.shape[1], 2))

# Sync the memmap array to disk
output.flush()

# Measure the end time
end_time = time.time()

# Measure memory usage after the process
final_memory = memory_usage(max_usage=True)

# Calculate runtime and memory usage
total_runtime = end_time - start_time
total_memory_usage = final_memory - initial_memory

# Load the results from the memmap file for further processing or analysis
output = np.memmap(output_filename_memmap, dtype='float64', shape=(num_tsne_maps, len(k_values)), mode='r')

# Example: Print all results
for idx, res in enumerate(output):
    print(f"Result for TSNE map {idx}: {res}")

# Print the total runtime and memory usage
print(f"Total runtime: {total_runtime} seconds")
print(f"Total memory usage: {total_memory_usage} MiB")
