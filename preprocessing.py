from scipy.io import mmread
import pandas as pd
import os
import numpy as np
import gzip
from joblib import dump, load
import time
from utils.slice_data_HVG import slice_data_HVG
from utils.data_pretreatment import preprocess_sparse_matrix
from utils.sample_row_ind import sampled_ind_matrix

## Load raw gene expression data, transpose and row-compress. Load metadata
data_path = os.path.join(
    os.path.dirname(os.getcwd()), 
    'thesis',
    'data',
    'matrix.mtx')
start_time = time.time()
data_sparse = mmread(data_path).transpose().tocsr()
print('Runtime - loading raw count mtx:', time.time() - start_time)
metadata = pd.read_csv(
    os.path.join(
        os.path.dirname(os.getcwd()), 
        'thesis',
        'data',
        '2097-Lungcancer_metadata.csv.gz'
    )
)
gene_data = pd.read_csv(
    os.path.join(
        os.path.dirname(os.getcwd()), 
        'thesis',
        'data',
        'genes.tsv'
    ),
    sep='\t',
)
print('Shape of the sparse data matrix is', data_sparse.shape)
print(metadata.head())
print('Unique Cell Types in metadata:', pd.unique(metadata['CellType']))
print('Shape of the gene data matrix is', gene_data.shape)
print(gene_data.head())
print('Are gene_data columns identical?',
    np.array_equal(gene_data.iloc[:, 0].values, gene_data.iloc[:, 1].values))


## Quality Control (QC) - cells
# Number of UMIs (Unique Molecular Identifiers)/total counts per cell
num_UMIs = metadata['nUMI'].values

# Number of genes detected per cell
num_genes = metadata['nGene'].values

# Fraction of mitochondrial counts per cell
# Assuming mitochondrial genes are labeled with 'MT-' prefix in the gene names
mito_genes = [gene for gene in gene_data.iloc[:, 0] if gene.startswith('MT-')]
mito_gene_indices = [i for i, gene in enumerate(gene_data.iloc[:, 0]) if gene in mito_genes]
mito_counts = np.array(data_sparse[:, mito_gene_indices].sum(axis=1)).flatten()
mito_fraction = mito_counts / num_UMIs

# Filter cells based on QC criteria
UMI_cutoff = 1000
gene_cutoff = 500
mito_cutoff = 0.2
qc_mask = (num_UMIs >= UMI_cutoff) & (num_genes >= gene_cutoff) & (mito_fraction <= 0.2)
data_sparse_qc = data_sparse[qc_mask]
metadata_qc = metadata[qc_mask]

print('Shape of the sparse data matrix after cell QC is', data_sparse_qc.shape)
print('Removal of:', 100*(1 - data_sparse_qc.shape[0]/data_sparse.shape[0]), '% of cells')
print('Shape of the metadata after cell QC is', metadata_qc.shape)

## Downsample data
# Generate row indices
ind_rows_downsample = sampled_ind_matrix(
    metadata = metadata_qc,
    nbr_samples = 5000,
    col_names=['CellFromTumor', 'PatientNumber', 'TumorType', 'TumorSite', 'CellType'])
print(ind_rows_downsample[:10])
print('Length of sampled indices:', len(ind_rows_downsample))

downsampled_sparse_data = data_sparse[ind_rows_downsample] #indexing relative to the original data
metadata_sampled = metadata.iloc[ind_rows_downsample, ]
print('Shape of Downsampled gene expression data: ', downsampled_sparse_data.shape)
print('Shape of Metadata: ', metadata_sampled.shape)
print(np.array_equal(metadata_sampled.index.values, ind_rows_downsample))

## Feature selection
# Select 0.1% HVG (mean-variance correction)
data_sp_csr_HVG = slice_data_HVG(downsampled_sparse_data, perc_top_genes=0.1)
print(data_sp_csr_HVG.shape)
print(data_sp_csr_HVG.__class__)