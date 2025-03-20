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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load raw gene expression data, transpose and compress. Load metadata and gene description data."""
    start_time = time.time()
    input_sparse_data = mmread(data_path).transpose().tocsr()
    logging.info(f'Runtime - loading raw count mtx: {time.time() - start_time}')
    
    # Load only necessary columns from metadata
    needed_columns = ['nUMI', 'nGene'] + grouping_columns
    metadata = pd.read_csv(metadata_path, 
                           usecols=needed_columns,
                           chunksize=10000)
    
    # Read gene data in chunks if needed
    gene_data = pd.read_csv(
        gene_data_path, 
        sep='\t',
        usecols=[0],
        chunksize=10000
    )
    gene_data = pd.concat(gene_data, ignore_index=True)
    
    logging.info(f'Shape of the sparse data matrix: {input_sparse_data.shape}')
    logging.info(f'Shape of the metadata data matrix: {metadata.shape}')
    logging.info(f'Metadata head: {metadata.head()}')
    logging.info(f'Shape of the gene data matrix: {gene_data.shape}')
    logging.info(f'Gene data head: {gene_data.head()}')
    
    return input_sparse_data, metadata, gene_data

def quality_control(data_sparse, metadata, gene_data):
    """Perform cell quality control on the raw input data."""
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
    metadata_qc.reset_index(drop=True, inplace=True)
    
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
    output_HVG = slice_data_HVG(downsampled_sparse_data, perc_top_genes=0.1)
    indices_HVG = output_HVG[1]
    logging.info(f'Number of identified highly variable genes: {sum(indices_HVG)}')
    return indices_HVG

def normLogTransformScale(data_sp_csr_HVG, HVG_indices, scale=True):
    """Pooling-based cell normalize, log-transformation, selection of the HVG, and z-score scaling."""
    normLogTransformScale_data = preprocess_sparse_matrix(data_sp_csr_HVG, HVG_indices, scale=scale)
    normLogTransformScale_data_name = 'LogNormalizedScaledData'
    logging.info(f'Shape of data after log-normalization and scaling: {normLogTransformScale_data.shape}')
    logging.info(f'Class of data after log-normalization and scaling: {normLogTransformScale_data.__class__}')
    return normLogTransformScale_data


def save_data(data, metadata=np.array([]), save_data_path=None, save_metadata_path=None):
    """Save data to file."""
    # Ensure the directory exists
    if save_data_path:
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
    else:
        logging.info('No save path provided for data. Data will be saved to the default directory.')
        save_data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'preprocessed_data.pkl.gz')
    
    if save_metadata_path:
        os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)
    
    try:
        # Save the data matrix with gzip compression
        dump(data, save_data_path, compress=('gzip', 3))
        data_name = getattr(data, 'name', 'preprocessed_data')
        logging.info(f'Data matrix {data_name} saved to {save_data_path}')
        
        # Save the metadata
        if len(metadata)!= 0 and save_metadata_path is not None:
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
    save_data(downsampled_sparse_data, metadata_sampled, save_sparse_data_path, save_metadata_path)
    indices_HVG_genes = feature_selection(downsampled_sparse_data)
    logNormalized_HVG_subset = normLogTransformScale(downsampled_sparse_data, indices_HVG_genes, scale=False)
    save_data(logNormalized_HVG_subset,
              save_data_path=os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'logNormalized_HVG_subset.pkl.gz'))

if __name__ == "__main__":
    main()