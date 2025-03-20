import os
import time
import numpy as np
import pandas as pd
from scipy.io import mmread
from memory_profiler import profile
import logging
import psutil
import gc
from scipy.sparse import csr_matrix

# Configure logging and paths
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', 'matrix.mtx')
metadata_path = os.path.join(os.path.dirname(os.getcwd()), 'thesis', 'data', '2097-Lungcancer_metadata.csv.gz')

def get_process_memory():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@profile
def method1_chunked_load_and_qc():
    """Method 1: Load and process matrix in chunks"""
    start_time = time.time()
    initial_memory = get_process_memory()
    
    # First read matrix dimensions
    with open(data_path, 'rb') as f:
        rows, cols, entries = map(int, f.readline().split())
        logging.info(f"Matrix dimensions: {rows} x {cols}")
    
    # Initialize lists to store QC'd chunks
    chunk_size = 20000
    qc_chunks = []
    total_cells_processed = 0
    
    # Load the entire matrix once
    sparse_matrix = mmread(data_path).transpose().tocsr()
    metadata = pd.read_csv(metadata_path)
    
    # Process matrix in chunks
    for chunk_start in range(0, rows, chunk_size):
        chunk_time = time.time()
        chunk_end = min(chunk_start + chunk_size, rows)
        
        # Slice the chunk
        chunk_sparse = sparse_matrix[chunk_start:chunk_end]
        chunk_metadata = metadata.iloc[chunk_start:chunk_end]
        
        # Apply QC to chunk
        chunk_UMIs = chunk_metadata['nUMI'].values
        chunk_genes = chunk_metadata['nGene'].values
        chunk_mask = (chunk_UMIs >= 500) & (chunk_genes >= 200)
        
        # Store QC'd chunk
        if np.any(chunk_mask):
            qc_chunks.append(chunk_sparse[chunk_mask])
            total_cells_processed += np.sum(chunk_mask)
        
        logging.info(f"Processed chunk {chunk_start}:{chunk_end} in {time.time() - chunk_time:.2f}s")
        logging.info(f"Cells passing QC in chunk: {np.sum(chunk_mask)}/{len(chunk_mask)}")
        gc.collect()
    
    # Concatenate all QC'd chunks
    if qc_chunks:
        final_matrix = csr_matrix.vstack(qc_chunks)
    
    final_memory = get_process_memory()
    logging.info(f"\nMethod 1 complete:")
    logging.info(f"Total time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Peak memory: {final_memory - initial_memory:.2f} MB")
    logging.info(f"Total cells passing QC: {total_cells_processed}/{rows}")
    
    return final_matrix

@profile
def method2_full_qc():
    """Method 2: Load full matrix + full QC"""
    start_time = time.time()
    initial_memory = get_process_memory()
    
    # 1. Load full matrix
    logging.info("Loading full sparse matrix...")
    sparse_matrix = mmread(data_path).transpose().tocsr()
    metadata = pd.read_csv(metadata_path)
    
    # 2. Process QC on full dataset
    UMIs = metadata['nUMI'].values
    genes = metadata['nGene'].values
    
    # QC calculations
    qc_mask = (UMIs >= 500) & (genes >= 200)
    final_matrix = sparse_matrix[qc_mask]
    
    gc.collect()
    
    final_memory = get_process_memory()
    logging.info(f"\nMethod 2 complete:")
    logging.info(f"Total time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Peak memory: {final_memory - initial_memory:.2f} MB")
    logging.info(f"Cells passing QC: {np.sum(qc_mask)}/{len(qc_mask)}")
    
    return final_matrix

def compare_methods():
    """Compare both methods"""
    logging.info("Starting comparison of methods...")
    
    logging.info("\nTesting Method 1: Chunked Loading and QC")
    final_matrix_1 = method1_chunked_load_and_qc()
    gc.collect()
    
    logging.info("\nTesting Method 2: Full Loading and QC")
    final_matrix_2 = method2_full_qc()
    gc.collect()
    
    # Verify results match
    logging.info("\nComparing results:")
    logging.info(f"Method 1 final shape: {final_matrix_1.shape}")
    logging.info(f"Method 2 final shape: {final_matrix_2.shape}")
    
    # Ensure the results are the same
    assert np.array_equal(final_matrix_1.toarray(), final_matrix_2.toarray()), "Results do not match!"

if __name__ == "__main__":
    compare_methods()