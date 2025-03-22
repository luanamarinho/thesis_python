import os
import time
import numpy as np
import pandas as pd
from scipy.io import mmread
from joblib import dump
from utils.slice_data_HVG import slice_data_HVG
from utils.sample_row_ind import sampled_ind_matrix
import logging
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.3'
from utils.data_pretreatment import preprocess_sparse_matrix
from memory_profiler import profile
import gc
from joblib import Memory
import psutil 
import scipy.sparse

# Global variables
data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'matrix.mtx')
metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', '2097-Lungcancer_metadata.csv.gz')
gene_data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'genes.tsv')
grouping_columns = ['CellFromTumor', 'PatientNumber', 'TumorType', 'TumorSite', 'CellType']
umi_cutoff = 1000
gene_cutoff = 500
mito_cutoff = 0.2
max_nbr_samples = 1000
seed = 42
save_metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'downsampled_metadata.csv')
file_sparse = f'downsampled_{max_nbr_samples}_sparse_gzip.pkl.gz'
save_sparse_data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', file_sparse)

# Caching
cache_dir = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'cache')
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, mmap_mode=None, verbose=0)

# Setting logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@profile
@memory.cache
def load_data():
    """Load raw gene expression data with memory tracking."""
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss
    
    try:
        input_sparse_data = mmread(data_path).transpose().tocsr()
        logging.info(f'Runtime - loading raw count mtx: {time.time() - start_time}')
        
        needed_columns = ['nUMI', 'nGene'] + grouping_columns
        metadata = pd.read_csv(metadata_path, usecols=needed_columns)
        
        gene_data = pd.read_csv(
            gene_data_path, 
            sep='\t',
            usecols=[0]
        )
        
        logging.info(f'Shape of the sparse data matrix: {input_sparse_data.shape}')
        logging.info(f'Memory usage: {(psutil.Process().memory_info().rss - initial_memory) / 1024 / 1024:.2f} MB')
        
        return input_sparse_data, metadata, gene_data
    finally:
        gc.collect()

@profile
@memory.cache
def quality_control(data_sparse, metadata, gene_data):
    """Perform cell quality control with memory tracking."""
    initial_memory = psutil.Process().memory_info().rss
    
    try:
        num_UMIs = metadata['nUMI'].values
        num_genes = metadata['nGene'].values
        
        mito_genes = [gene for gene in gene_data.iloc[:, 0] if gene.startswith('MT-')]
        if len(mito_genes) > 0:
            mito_gene_indices = [i for i, gene in enumerate(gene_data.iloc[:, 0]) if gene in mito_genes]
            mito_counts = np.array(data_sparse[:, mito_gene_indices].sum(axis=1)).flatten()
            mito_fraction = mito_counts / num_UMIs
            del mito_counts
        else:
            logging.info('No mitochondrial genes found in the gene data')
            mito_fraction = np.zeros(data_sparse.shape[0])
        
        qc_mask = (num_UMIs >= umi_cutoff) & (num_genes >= gene_cutoff) & (mito_fraction <= mito_cutoff)
        data_sparse_qc = data_sparse[qc_mask]
        metadata_qc = metadata.loc[qc_mask].reset_index(drop=True)
        
        logging.info(f'Shape after QC: {data_sparse_qc.shape}')
        logging.info(f'Memory usage: {(psutil.Process().memory_info().rss - initial_memory) / 1024 / 1024:.2f} MB')
        
        return data_sparse_qc, metadata_qc
    finally:
        del qc_mask, mito_fraction
        gc.collect()

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
    metadata_sampled = metadata_qc.iloc[ind_rows_downsample]
    metadata_sampled.name = 'downsampled_metadata'
    
    logging.info(f'Shape of Downsampled gene expression data: {downsampled_sparse_data.shape}')
    logging.info(f'Shape of Metadata: {metadata_sampled.shape}')
    logging.info(f'Indices of metadata match expected sampled indices: {np.array_equal(metadata_sampled.index.values, ind_rows_downsample)}')
    
    return downsampled_sparse_data, metadata_sampled

def feature_selection(downsampled_sparse_data):
    """Select highly variable genes (HVG)."""
    output_HVG = slice_data_HVG(downsampled_sparse_data, perc_top_genes=0.1)
    indices_HVG = output_HVG[1]
    logging.info(f'Number of identified highly variable genes: {sum(indices_HVG)}')
    return indices_HVG

def normLogTransformScale(data_sp_csr_HVG, HVG_indices, scale=True):
    """Pooling-based cell normalize, log-transformation, selection of the HVG, and z-score scaling."""
    normLogTransformScale_data = preprocess_sparse_matrix(data_sp_csr_HVG, HVG_indices, scale=scale)
    logging.info(f'Shape of data after log-normalization and scaling: {normLogTransformScale_data.shape}')
    logging.info(f'Class of data after log-normalization and scaling: {normLogTransformScale_data.__class__}')
    return normLogTransformScale_data

@profile
def save_data(data, metadata=np.array([]), save_data_path=None, save_metadata_path=None):
    """Save data in efficient format with memory tracking."""
    try:
        if save_data_path:
            os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
            
            # npz format for sparse matrices
            if scipy.sparse.issparse(data):
                save_path = save_data_path.replace('.pkl.gz', '.npz')
                scipy.sparse.save_npz(save_path, data)
                logging.info(f'Sparse matrix saved to {save_path}')
            else:
                dump(data, save_data_path, compress=('gzip', 3))
                logging.info(f'Data saved to {save_data_path}')
        
        if len(metadata) != 0 and save_metadata_path is not None:
            os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)
            metadata.to_csv(save_metadata_path, index=False)
            logging.info(f'Metadata saved to {save_metadata_path}')
            
    except Exception as e:
        logging.error(f'Error saving data: {e}')
    finally:
        gc.collect()

@profile
def main():
    """Main function with memory tracking."""
    initial_memory = psutil.Process().memory_info().rss
    
    try:
        # Load and QC
        data_sparse, metadata, gene_data = load_data()
        gc.collect()
        
        data_sparse_qc, metadata_qc = quality_control(data_sparse, metadata, gene_data)
        del data_sparse, metadata, gene_data
        gc.collect()
        
        # Downsample
        downsampled_sparse_data, metadata_sampled = downsample_data(data_sparse_qc, metadata_qc)
        del data_sparse_qc, metadata_qc
        gc.collect()
        
        # Save downsampled data
        save_data(downsampled_sparse_data, metadata_sampled, 
                 save_sparse_data_path, save_metadata_path)
        
        # Feature selection and normalization
        indices_HVG_genes = feature_selection(downsampled_sparse_data)
        logNormalized_HVG_subset = normLogTransformScale(
            downsampled_sparse_data, indices_HVG_genes, scale=False)
        del downsampled_sparse_data
        gc.collect()
        
        # Save final results
        fname = f'logNormalized_HVG_subset_{max_nbr_samples}_samples.pkl'
        final_save_path = os.path.join(os.path.dirname(os.getcwd()), 
                                     'thesis', 'data', 
                                     fname)
        save_data(logNormalized_HVG_subset, save_data_path=final_save_path)
        
    finally:
        final_memory = psutil.Process().memory_info().rss
        logging.info(f'Total memory change: {(final_memory - initial_memory) / 1024 / 1024:.2f} MB')
        gc.collect()

if __name__ == "__main__":
    main()