# double checking stress outtput for some tsne maps
import os
import numpy as np
from joblib import load
from utils.compute_trust_stress_chunked import compute_stress_trust
import pandas as pd
import ast
import openTSNE

path_base = '~/Documents/data'
num_maps_tsne = int(990/2)

path_output = os.path.join(path_base, 'output_memmap_stress_real_0_990_momentum.dat')
expanded_path_output = os.path.expanduser(path_output)
output_stress = np.memmap(expanded_path_output, shape=(num_maps_tsne, 1), dtype='float64', mode='r')

dist_X = load(os.path.join("C:/Users/luana/Documents/data","dist_X_5000.joblib"), mmap_mode='r')
data_tsne = load(os.path.join(f"C:/Users/luana/Documents/data","df_tsne_momentum.joblib"), mmap_mode='r')
data_tsne = data_tsne.values


first_indices_maps = [i for i in range(0, 990, 2)]
maps_to_check_stress = output_stress >= 1
indices_of_maps_to_check = np.arange(num_maps_tsne)[maps_to_check_stress.flatten()]
#first_indices_of_maps_to_check = first_indices_maps[indices_of_maps_to_check]
first_indices_of_maps_to_check = [first_indices_maps[i] for i in indices_of_maps_to_check]

stress_list = []

for index in first_indices_of_maps_to_check:
    tsne_map = data_tsne[:, index:index+2]  # Extract the TSNE map (2 columns)
    #result = compute_trustworthiness(dist_X, tsne_map, k)
    #output[(index - start_index) // 2] = result
    trust, stress,_ = compute_stress_trust(dist_X, tsne_map, [30,300])
    stress_list.append(stress)

output_stress[maps_to_check_stress.flatten()].flatten().shape
(output_stress[maps_to_check_stress.flatten()].flatten() == stress_list).all()




