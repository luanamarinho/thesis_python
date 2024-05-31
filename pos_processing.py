from joblib import load
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import os

output = load('/home/luana/workspace/output/thesis/perp5/pipeline_multiples_perp5_0-60.joblib')

len(output)
len(output[0]) # contains 60 pipeline outputs
output[0][0] # 0) comb, 1) tSNE map, 2) tsne runtime excluding affinity calculation, 3) KL divergence, 4) index of pipeline in the output


combinations = [tuple[0] for tuple in output[0]]
affinity_runtime_min = output[1]/60 # 5.6
tsne_runtime_min = [tuple[2]/60 for tuple in output[0]]
avg_tsne_runtime_min = np.mean(tsne_runtime_min)/60
KL_divergences = [tuple[3] for tuple in output[0]]
Pipeline_indices = [tuple[4] for tuple in output[0]]
tsne_maps = [tuple[2]/60 for tuple in output[0]]

modified_tsne_dfs = []
for tuple in output[0]:
    combination = tuple[0]
    df_tsne = tuple[1]
    df_tsne.columns = [f'tSNE1_{combination}', f'tSNE2_{combination}']
    modified_tsne_dfs.append(df_tsne)

merged_tsnes = pd.concat(modified_tsne_dfs, axis=1)

import itertools

output_path = '/home/luana/workspace/output/thesis'
output_folders = os.listdir(output_path)
output_files = [os.listdir(os.path.join(output_path, folder)) for folder in output_folders]
output_files_flat = [file for file in itertools.chain(*output_files) if file != 'README']

#pipelines_merged = [load(file) for file in output_files_flat]


    
from joblib import load
import pandas as pd

def extract_data_from_file(output_file_path):
  
  output = load(output_file_path)
  
  combinations = [tuple[0] for tuple in output[0]]
  affinity_runtime_min = [output[1]/60 for tuple in output[0]]
  tsne_runtime_min = [tuple[2]/60 for tuple in output[0]]
  KL_divergences = [tuple[3] for tuple in output[0]]
  Pipeline_indices = [tuple[4] for tuple in output[0]]

  modified_tsne_dfs = []
  for tuple in output[0]:
    combination = tuple[0]
    df_tsne = tuple[1]
    df_tsne.columns = [f'tSNE1_{combination}', f'tSNE2_{combination}']
    modified_tsne_dfs.append(df_tsne)

  merged_tsnes_from_file = pd.concat(modified_tsne_dfs, axis=1)

  perplexity = [comb[0] for comb in combinations] # 5
  early_exaggeration = [comb[1] for comb in combinations]
  initial_momentum = [comb[2] for comb in combinations]
  final_momentum = [comb[3] for comb in combinations]
  theta = [comb[3] for comb in combinations]

  pipeline_metrics = pd.DataFrame({'Pipeline_index': Pipeline_indices,
                                 'Combination' : combinations,
                                 'Perplexity': perplexity,
                                 'Early_exaggeration': early_exaggeration,
                                 'Initial_momentum': initial_momentum,
                                 'Final_momentum': final_momentum,
                                 'Theta': theta,
                                 'tSNE_runtime_min': tsne_runtime_min,
                                 'Affinity_runtime_min': affinity_runtime_min,
                                 'KL_divergence': KL_divergences
                                 })

  return(merged_tsnes_from_file, pipeline_metrics)




out1 = extract_data_from_file('/home/luana/workspace/output/thesis/perp5/pipeline_multiples_perp5_0-60.joblib')
out1[0]
out1[0].shape
out1[1]
output = load('/home/luana/workspace/output/thesis/perp5/pipeline_multiples_perp5_0-60.joblib')
len(output[0])

import os
import itertools

def merge_output_from_folder(folder_relative_path = 'perp5'):
  output_files = os.listdir(os.path.join(output_path, folder_relative_path))
  tsne_maps_list = []
  pipeline_metrics_list = []
  for file in output_files:
    if file != 'README':
      full_file_path = os.path.join(output_path, folder_relative_path, file)
      data_from_file = extract_data_from_file(full_file_path)
      tsne_maps_list.append(data_from_file[0])
      pipeline_metrics_list.append(data_from_file[1])
     

  #tsne_maps_list = [tsne_df for tsne_df in itertools.chain(*tsne_maps_list)]
  #pipeline_metrics_list = [metrics_df for metrics_df in itertools.chain(*pipeline_metrics_list)]
  return(tsne_maps_list, pipeline_metrics_list)


out = merge_output_from_folder()
len(out)
len(out[0]) #4 t
len(out[1])
   





output_path = '/home/luana/workspace/output/thesis'
folders = os.listdir(output_path)

all_tsne_maps = []
all_pipeline_metrics = []

for folder in folders:
   output_folder = merge_output_from_folder(folder)
   all_tsne_maps.append(output_folder[0])
   all_pipeline_metrics.append(output_folder[0])
   







folder_path = '/home/luana/workspace/output/thesis/perp45'
files = os.listdir(folder_path)
for file in files:
   if file != 'README':
    file_full = os.path.join(folder_path, file)
    out = load(file_full)
    print(len(out[0]))
































## Plots______________________________________________

perplexity = [comb[0] for comb in combinations] # 5
early_exaggeration = [comb[1] for comb in combinations]
initial_momentum = [comb[2] for comb in combinations]
final_momentum = [comb[3] for comb in combinations]
theta = [comb[3] for comb in combinations]

pipeline_metrics = pd.DataFrame({'Pipeline_index': Pipeline_indices,
                                 'Combination' : combinations,
                                 'Perplexity': perplexity,
                                 'Early_exaggeration': early_exaggeration,
                                 'Initial_momentum': initial_momentum,
                                 'Final_momentum': final_momentum,
                                 'Theta': theta,
                                 'tSNE_runtime_min': tsne_runtime_min,
                                 'KL_divergence': KL_divergences
                                 })

df_pipelines = pipeline_metrics.iloc[:,3:] # Removing perplexity as well
#pairwise relationships and patterns between all variables
sns.pairplot(df_pipelines)
plt.show()

# Compute the Pearson correlation coefficients and adjusted p-values
corr_matrix = df_pipelines.corr()
pvals = np.zeros((df_pipelines.shape[1], df_pipelines.shape[1]))
for i in range(df_pipelines.shape[1]):
    for j in range(df_pipelines.shape[1]):
        if i != j:
            _, pvals[i, j] = pearsonr(df_pipelines.iloc[:, i], df_pipelines.iloc[:, j])

# Flatten the p-values matrix to apply multiple testing correction
pvals_flat = pvals.flatten()

# Perform multiple testing correction using Benjamini-Hochberg
_, pvals_corrected_flat, _, _ = multipletests(pvals_flat, alpha=0.05, method='fdr_bh')

# Reshape the corrected p-values back to a matrix
pvals_corrected = pvals_corrected_flat.reshape(pvals.shape)

# Define significance level
significance_level = 0.05

# Mask for significant correlations
mask = pvals_corrected >= significance_level

# Apply the mask to the correlation matrix
corr_matrix_significant = corr_matrix.mask(mask)

# Create a mask for the upper triangle and diagonal
mask_upper = np.triu(np.ones(corr_matrix.shape), k=0).astype(bool)

# Apply the mask to show only the upper triangle
corr_matrix_significant = corr_matrix_significant.mask(mask_upper)

# Plot the filtered correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_significant, annot=True, cmap='coolwarm', cbar_kws={"shrink": .8}, vmin=-1, vmax=1)
plt.title('Correlation Matrix with Only Significant Correlations (Adjusted P-values)')
plt.show()


# Relationship between KL and the parameters
parameters = ['Early_exaggeration', 'Initial_momentum','Final_momentum', 'Theta']
df = pipeline_metrics[['KL_divergence', 'Early_exaggeration', 'Initial_momentum','Final_momentum', 'Theta']]
sns.pairplot(df)
plt.show()

# Runtime and the parameters
df = pipeline_metrics[['tSNE_runtime_min', 'Early_exaggeration', 'Initial_momentum','Final_momentum', 'Theta']]
sns.pairplot(df)
plt.show()

# Scatter plot of runtime, KL divergence and theta
df = pipeline_metrics[['tSNE_runtime_min', 'KL_divergence', 'Theta']]
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Theta',
    y='tSNE_runtime_min',
    size='KL_divergence',
    sizes=(20, 2000),  # Adjust this to scale marker sizes appropriately
    alpha=0.6,
    legend=False
)

plt.xlabel('Theta')
plt.ylabel('Runtime')
plt.title('Scatter Plot of Theta vs. Runtime with KL_divergence as Marker Size')
plt.show()