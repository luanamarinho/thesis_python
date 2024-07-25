import pandas as pd
import numpy as np
import time
from memory_profiler import memory_usage
from utils.compute_trustworthiness_chunked import compute_trustworthiness
from utils.compute_trust_stress_chunked_norm import compute_stress_trust
from joblib import Parallel, delayed, load
import os
import argparse

def run_compute_metrics(folder = "C:/Users/luana/Documents/data", start_index = 0, end_index = None, k = [30,300], toy_data = False):
  # Load the input distance matrix and the tsne maps
  if not toy_data:
    dist_X = load(os.path.join(folder,"dist_X_5000.joblib"), mmap_mode='r')
    data_tsne = load(os.path.join(folder,"df_tsne_momentum.joblib"), mmap_mode='r')
  elif toy_data:
    dist_X = load(os.path.join('output',"dist_X_toy.joblib"), mmap_mode='r')
    data_tsne = load(os.path.join('output',"data_mds_toy.joblib"), mmap_mode='r')
  
  if end_index is None:
    end_index = data_tsne.shape[1]

  # Pre-allocate a np memmap output array
  num_tsne_maps = (end_index - start_index) // 2

  output_filename_trust = os.path.join(folder, f'output_memmap_trust_{"toy" if toy_data else "real"}_{start_index}_{end_index}_momentum.dat')
  output_filename_stress = os.path.join(folder, f'output_memmap_stress_{"toy" if toy_data else "real"}_{start_index}_{end_index}_momentum.dat')

  output_trust = np.memmap(output_filename_trust, dtype='float64', shape=(num_tsne_maps, len(k)), mode='w+')
  output_stress = np.memmap(output_filename_stress, dtype='float64', shape=(num_tsne_maps,), mode='w+')

  # Memory usage and time() before the execution
  initial_memory = memory_usage(max_usage=True)
  start_time = time.time()

  # Parallel processing with joblib 
  #Parallel(n_jobs=-1)(delayed(process_tsne_map)(i, output, data_tsne, dist_X, k, start_index) for i in range(start_index, end_index, 2))
  Parallel(n_jobs=-1)(delayed(process_tsne_map)(i, output_trust, output_stress, data_tsne, dist_X, k, start_index) for i in range(start_index, end_index, 2))

  # Sync memmap arrays to disk
  output_trust.flush()
  output_stress.flush()

  # End time and memory usage at the end of process
  end_time = time.time()
  final_memory = memory_usage(max_usage=True)

  # Compute runtime and memory usage
  total_runtime = end_time - start_time
  total_memory_usage = final_memory - initial_memory

# Save runtime and memory usage to file
  summary_filename = os.path.join(folder, 'summary_runtime_memory.txt')
  with open(summary_filename, 'w') as summary_file:
    summary_file.write(f"Total runtime: {total_runtime} seconds\n")
    summary_file.write(f"Total memory usage: {total_memory_usage} MiB\n")
  
  # Print the total runtime and memory usage
  print(f"Total runtime: {total_runtime} seconds")
  print(f"Total memory usage: {total_memory_usage} MiB")

  return output_trust, output_filename_trust, output_stress, output_filename_stress, num_tsne_maps, len(k) 


# Process each TSNE map
def process_tsne_map(index, output_trust, output_stress, data_tsne, dist_X, k, start_index):
  tsne_map = data_tsne[:, index:index+2]  # Extract the TSNE map (2 columns)
  #result = compute_trustworthiness(dist_X, tsne_map, k)
  #output[(index - start_index) // 2] = result
  trust, stress,_ = compute_stress_trust(dist_X, tsne_map, k)
  output_trust[(index - start_index) // 2] = trust
  output_stress[(index - start_index) // 2] = stress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compute metrics")
    parser.add_argument("--folder", type=str, default="C:/Users/luana/Documents/data", help="Folder containing input data")
    parser.add_argument("--start_index", type=int, default=0, help="Start index")
    parser.add_argument("--end_index", type=int, help="End index")
    parser.add_argument("--k", type=int, nargs='+', default=[30, 300], help="List of k values for trustworthiness")
    parser.add_argument("--toy_data", action='store_true', help="Use toy data if specified")

    args = parser.parse_args()

    run_compute_metrics(folder=args.folder, start_index=args.start_index, end_index=args.end_index, k=args.k, toy_data=args.toy_data)


# Load the results from the memmap file for further processing or analysis
#output = np.memmap(output_filename_memmap, dtype='float64', shape=(num_tsne_maps, len(k_values)), mode='r')

#Print all results
#for idx, res in enumerate(output):

#To explicitly set toy_data to True: python script_name.py --toy_data

