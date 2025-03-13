import os
import time
import numpy as np
import pandas as pd
from scipy.io import mmread
from joblib import dump
from utils.slice_data_HVG import slice_data_HVG
from utils.data_pretreatment import preprocess_sparse_matrix
from utils.sample_row_ind import sampled_ind_matrix
import logging

# Global variables
data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'matrix.mtx')
metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', '2097-Lungcancer_metadata.csv.gz')
gene_data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'genes.tsv')
grouping_columns = ['CellFromTumor', 'PatientNumber', 'TumorType', 'TumorSite', 'CellType']
umi_cutoff = 1000
gene_cutoff = 500
mito_cutoff = 0.2
max_nbr_samples = 5000
seed = 42
save_metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'downsampled_metadata.csv')
file_sparse = f'downsampled_{max_nbr_samples}_sparse_gzip.pkl.gz'
save_sparse_data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', file_sparse)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load raw gene expression data, transpose and compress. Load metadata and gene description data."""
    start_time = time.time()
    data_sparse = mmread(data_path).transpose().tocsr()
    logging.info(f'Runtime - loading raw count mtx: {time.time() - start_time}')
    
    metadata = pd.read_csv(metadata_path)
    gene_data = pd.read_csv(gene_data_path, sep='\t')
    
    logging.info(f'Shape of the sparse data matrix: {data_sparse.shape}')
    logging.info(f'Metadata head: {metadata.head()}')
    logging.info(f'Unique Cell Types in metadata: {pd.unique(metadata["CellType"])}')
    logging.info(f'Shape of the gene data matrix: {gene_data.shape}')
    logging.info(f'Gene data head: {gene_data.head()}')
    logging.info(f'Are gene_data columns identical? {np.array_equal(gene_data.iloc[:, 0].values, gene_data.iloc[:, 1].values)}')
    
    return data_sparse, metadata, gene_data

def quality_control(data_sparse, metadata, gene_data):
    """Perform cell quality control on the data."""
    num_UMIs = metadata['nUMI'].values
    num_genes = metadata['nGene'].values
    
    mito_genes = [gene for gene in gene_data.iloc[:, 0] if gene.startswith('MT-')]
    if len(mito_genes) > 0: 
        mito_gene_indices = [i for i, gene in enumerate(gene_data.iloc[:, 0]) if gene in mito_genes]
        mito_counts = np.array(data_sparse[:, mito_gene_indices].sum(axis=1)).flatten()
        mito_fraction = mito_counts / num_UMIs
    else:
        logging.info('No mitochondrial genes found in the gene data') 
        mito_fraction = np.zeros(data_sparse.shape[0])
    
    qc_mask = (num_UMIs >= umi_cutoff) & (num_genes >= gene_cutoff) & (mito_fraction <= mito_cutoff)
    data_sparse_qc = data_sparse[qc_mask]
    metadata_qc = metadata[qc_mask]
    
    logging.info(f'Shape of the sparse data matrix after cell QC: {data_sparse_qc.shape}')
    logging.info(f'Removal of: {100 * (1 - data_sparse_qc.shape[0] / data_sparse.shape[0])}% of cells')
    logging.info(f'Shape of the metadata after cell QC: {metadata_qc.shape}')
    
    return data_sparse_qc, metadata_qc

def downsample_data(data_sparse_qc, metadata_qc):
    """Downsample the data."""
    ind_rows_downsample = sampled_ind_matrix(
        metadata=metadata_qc,
        nbr_samples=max_nbr_samples,
        seed=seed,
        col_names=grouping_columns
    )
    
    logging.info(f'First 10 sampled indices: {ind_rows_downsample[:10]}')
    logging.info(f'Length of sampled indices: {len(ind_rows_downsample)}')
    
    downsampled_sparse_data = data_sparse_qc[ind_rows_downsample]
    metadata_sampled = metadata_qc.iloc[ind_rows_downsample, :]
    metadata_sampled.name = 'downsampled_metadata'
    
    logging.info(f'Shape of Downsampled gene expression data: {downsampled_sparse_data.shape}')
    logging.info(f'Shape of Metadata: {metadata_sampled.shape}')
    logging.info(f'Indices of metadata match computed indices: {np.array_equal(metadata_sampled.index.values, ind_rows_downsample)}')
    
    return downsampled_sparse_data, metadata_sampled

def feature_selection(downsampled_sparse_data):
    """Select highly variable genes (HVG)."""
    data_sp_csr_HVG = slice_data_HVG(downsampled_sparse_data, perc_top_genes=0.1)
    data_sp_csr_HVG.name = 'downsampled_sparse_data_HVG'
    logging.info(f'Shape of raw data after feature selection: {data_sp_csr_HVG.shape}')
    logging.info(f'Class of raw data after feature selection: {data_sp_csr_HVG.__class__}')
    return data_sp_csr_HVG

def normLogTransformScale(data_sp_csr_HVG):
    """Normalize, log-transform, and scale the data."""
    normLogTransformScale_data = preprocess_sparse_matrix(data_sp_csr_HVG)
    normLogTransformScale_data.name = 'normScaleLogTransform_data'
    logging.info(f'Shape of data after normalization, log transformation, and scaling: {normLogTransformScale_data.shape}')
    logging.info(f'Class of data after normalization,  log transformation, and scaling: {normLogTransformScale_data.__class__}')


def save_data(data, metadata, save_data_path, save_metadata_path):
    """Save the downsampled data and metadata to files."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)
    
    try:
        # Save the data matrix
        dump(data, save_data_path, compress=('gzip', 3))
        data_name = getattr(data, 'name', 'preprocessed_data')
        logging.info(f'Data matrix {data_name} saved to {save_data_path}')
        
        # Save the metadata
        metadata.to_csv(save_metadata_path, index=False)
        metadata_name = getattr(metadata, 'name', 'preprocessed_metadata')
        logging.info(f'Metadata {metadata_name} saved to {save_metadata_path}')
    except Exception as e:
        logging.error(f'Error saving data: {e}')

def main():
    """Main function to run the preprocessing workflow."""
    data_sparse, metadata, gene_data = load_data()
    data_sparse_qc, metadata_qc = quality_control(data_sparse, metadata, gene_data)
    downsampled_sparse_data, metadata_sampled = downsample_data(data_sparse_qc, metadata_qc)
    data_sp_csr_HVG = feature_selection(downsampled_sparse_data)
    save_data(data_sp_csr_HVG, metadata_sampled, save_sparse_data_path, save_metadata_path)

if __name__ == "__main__":
    main()