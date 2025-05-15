import os
import time
import numpy as np
import pandas as pd
from joblib import dump
from utils.slice_data_HVG import slice_data_HVG
from utils.sample_row_ind import sampled_ind_matrix
import logging
os.environ['R_HOME'] = '/usr/lib/R'
from utils.data_pretreatment import preprocess_sparse_matrix
import gc
import psutil
import tarfile
import tempfile
import os
from scipy.io import mmread
import scipy.sparse
from scanpy import read_mtx

# Global variables
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'matrix.mtx')
metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'data', '2097-Lungcancer_metadata.csv')
gene_data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'genes.tsv')
# tar_gz_path = os.path.join(os.path.dirname(os.getcwd()), 'data', '2096-Lungcancer_counts.tar.gz')
grouping_columns = ['CellFromTumor', 'PatientNumber', 'TumorType', 'TumorSite', 'CellType']
umi_cutoff = 1000
gene_cutoff = 500
mito_cutoff = 0.2
max_nbr_samples = 5000
seed = 42
scale = True
save_metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'downsampled_metadata.csv')
file_sparse = f'downsampled_{max_nbr_samples}_sparse_gzip.pkl.gz'
save_sparse_data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', file_sparse)

# Setting logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

def extract_compressed_file_name(tar_gz_path):
    """Extracts compressed files from the file path."""

    temp_files = {
        'matrix': None,
        'genes': None,
        'barcodes': None
    }

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("matrix.mtx"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mtx") as tmp:
                    tmp.write(tar.extractfile(member).read())
                    temp_files['matrix'] = tmp.name
            elif member.name.endswith("genes.tsv"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as tmp:
                    tmp.write(tar.extractfile(member).read())
                    temp_files['genes'] = tmp.name
            elif member.name.endswith("barcodes.tsv"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as tmp:
                    tmp.write(tar.extractfile(member).read())
                    temp_files['barcodes'] = tmp.name


def load_metadata(metadata_path):
    """Load metadata."""
    logging.info(f"Loading metadata from {metadata_path}")
    start_time = time.time()
    try:
        needed_columns = ['nUMI', 'nGene'] + grouping_columns
        metadata = pd.read_csv(metadata_path, usecols=needed_columns)
        logging.info(f'Runtime - loading metadata: {time.time() - start_time:.2f} seconds')
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata: {str(e)}")
        raise

def load_gene_data(gene_data_path):
    """Load gene data."""
    logging.info(f"Loading gene data from {gene_data_path}")

    start_time = time.time()
    try:
        gene_data = pd.read_csv(gene_data_path, sep='\t', usecols=[0])
        logging.info(f'Runtime - loading gene data: {time.time() - start_time:.2f} seconds')
        return gene_data
    except Exception as e:
        logging.error(f"Error loading gene data: {str(e)}")
        raise

def quality_control(data_sparse, metadata, gene_data):
    """Perform cell quality control with memory tracking."""
    initial_memory = psutil.Process().memory_info().rss
    
    try:
        num_UMIs = metadata['nUMI'].values
        num_genes = metadata['nGene'].values
        
        # mito_genes = [gene for gene in gene_data.iloc[:, 0] if gene.startswith('MT-')]
        mito_genes = gene_data[gene_data.iloc[:, 0].str.startswith('MT-')].iloc[:, 0].tolist()
        if len(mito_genes) > 0:
            # mito_gene_indices = [i for i, gene in enumerate(gene_data.iloc[:, 0]) if gene in mito_genes]
            mito_gene_indices = gene_data[gene_data.iloc[:, 0].isin(mito_genes)].index.tolist()
            mito_counts = np.array(data_sparse[:, mito_gene_indices].sum(axis=1)).flatten()
            mito_fraction = mito_counts / num_UMIs
            del mito_counts
        else:
            logging.info('No mitochondrial genes found in the gene data')
            mito_fraction = np.zeros(data_sparse.shape[0])
        
        qc_mask = (num_UMIs >= umi_cutoff) & (num_genes >= gene_cutoff) & (mito_fraction <= mito_cutoff)
        data_sparse_qc = data_sparse[qc_mask]
        metadata_qc = metadata.loc[qc_mask].reset_index(drop=True)
        
        # logging.info(f'Shape after QC: {data_sparse_qc.shape}')
        logging.info(f'Memory usage: {(psutil.Process().memory_info().rss - initial_memory) / 1024 / 1024:.2f} MB')
        
        return data_sparse_qc, metadata_qc
    finally:
        del qc_mask, mito_fraction
        gc.collect()

def downsample_data(data_sparse_qc, metadata_qc, max_nbr_samples=max_nbr_samples):
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

def main():
    """Main function."""
    try:
        logging.info("Starting preprocessing pipeline")

        # tmp_files = extract_compressed_file_name(tar_gz_path=tar_gz_path)
        # data_path = tmp_files['matrix']
        # gene_data_path = tmp_files['genes']
        # Load data
        logging.info("Step 1: Loading data")
        
        # Load sparse matrix directly in main
        logging.info(f"Loading sparse matrix from {data_path}")
        matrix_start = time.time()
        data_sparse = mmread(data_path).T.tocsr()
        logging.info(f'Runtime - loading sparse matrix: {time.time() - matrix_start:.2f} seconds')
        logging.info(f'Shape of sparse matrix: {data_sparse.shape}')
        logging.info(f'Class of sparse matrix: {data_sparse.__class__}')

        # Load other data
        metadata = load_metadata(metadata_path)
        logging.info(f'Shape of metadata: {metadata.shape}')
        gene_data = load_gene_data(gene_data_path)
        logging.info(f'Shape of gene data: {gene_data.shape}')
        
        gc.collect()
        
        logging.info("Step 2: Performing quality control")
        data_sparse_qc, metadata_qc = quality_control(data_sparse, metadata, gene_data)
        logging.info(f'Shape or count data after QC: {data_sparse_qc.shape}')
        logging.info(f'Shape of metadata after QC: {metadata_qc.shape}')
        del data_sparse, metadata, gene_data
        gc.collect()
        
        # Downsample
        logging.info("Step 3: Downsampling data")
        downsampled_sparse_data, metadata_sampled = downsample_data(data_sparse_qc, metadata_qc)
        del data_sparse_qc, metadata_qc
        gc.collect()
        
        # Save downsampled data
        logging.info("Step 4: Saving downsampled data")
        save_data(downsampled_sparse_data, metadata_sampled, 
                 save_sparse_data_path, save_metadata_path)
        
        # Feature selection and normalization
        logging.info("Step 5: Performing feature selection")
        indices_HVG_genes = feature_selection(downsampled_sparse_data)
        
        logging.info("Step 6: Normalizing and scaling data")
        logNormalized_HVG_subset = normLogTransformScale(
            downsampled_sparse_data, indices_HVG_genes, scale=scale)
        del downsampled_sparse_data
        gc.collect()
        
        # Save final results
        logging.info("Step 7: Saving final results")
        fname = f'logNormalized_HVG_subset_{max_nbr_samples}_samples_scaled_{scale}.pkl'
        final_save_path = os.path.join(os.path.dirname(os.getcwd()), 
                                     'data', 
                                     fname)
        save_data(logNormalized_HVG_subset, save_data_path=final_save_path)
        
        logging.info("Preprocessing pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        raise