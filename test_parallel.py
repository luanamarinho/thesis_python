import numpy as np
import h5py
from joblib import load, dump, Parallel, delayed
from utils.trustworthiness_chunks_1k import trustworthiness_chunks
from time import time
import sys
import os
import pandas as pd

# Example: creating a large distance matrix
dist_input_filepath = "C:/Users/luana/Documents/data/dist_X.joblib"

starttime = time()
X = load(dist_input_filepath) #71.71637177467346
load_time = time() - starttime
sys.getsizeof(X) #12780807328
del X

starttime = time()
data = load(dist_input_filepath, mmap_mode='r')
load_time_mmap = time() - starttime # 0.0525
sys.getsizeof(data) #160


chunk_size = int(1e3)
slices = [slice(start, start + chunk_size) for start in range(0, len(data), chunk_size)]
len(slices)       

def slow_mean_write_output(data, sl, output, idx):
  res_ = data[sl].mean()
  print("[Worker %d] Mean for slice %d is %f" % (os.getpid(), idx, res_))
  output[idx] = res_

folder =  "C:/Users/luana/Documents/data"
output_filename_memmap = os.path.join(folder, 'output_memmap')

#Pre-allocate a writable shared memory map as a container for the results of the parallel computation.
output = np.memmap(output_filename_memmap, dtype=data.dtype,
                   shape=len(slices), mode='w+')
sys.getsizeof(output) #144

Parallel(n_jobs=2)(delayed(slow_mean_write_output)(data, sl, output, idx)
                   for idx, sl in enumerate(slices))
len(output)
output[0]

df_tsne_filepath = "output/df_tsne_unique.csv"
df_tsne = pd.read_csv(df_tsne_filepath, compression='gzip')
df_tsne.shape
sys.getsizeof(df_tsne) #349817604
tsne_npy = df_tsne.to_numpy()
dump(tsne_npy, os.path.join(folder,"df_tsne.joblib"))

data_tsne = load(os.path.join(folder,"df_tsne.joblib"), mmap_mode='r')
data_tsne.shape
sys.getsizeof(data_tsne) #160

tsne_map = data_tsne[:,:2]
sys.getsizeof(tsne_map) # 160
tsne_map











# Creating the HDF5 file
with h5py.File('distance_matrix.h5', 'w') as f:
    f.create_dataset('distance_matrix', data=distance_matrix)



