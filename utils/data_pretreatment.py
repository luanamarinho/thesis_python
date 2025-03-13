import numpy as np
from sklearn.preprocessing import normalize, StandardScaler

def preprocess_sparse_matrix(data_sp_csr_HVG, normalization='l1'):
    """
    Preprocesses a sparse matrix by normalizing, log-transforming, and scaling its values.
    
    Parameters:
        data_sparse (scipy.sparse.csr_matrix): Input sparse matrix to be preprocessed.
        normalization (str): Type of normalization to be applied. Default is 'l1'. {'l1', 'l2', 'max'}
        
    Returns:
        numpy.ndarray: Preprocessed dense matrix.
    """
    data_dense = data_sp_csr_HVG.toarray()
    data_normalized = normalize(data_dense, norm=normalization, axis=1)

    # Log transform the data
    data_log_transformed = np.log1p(data_normalized)  # Apply log(1+x) transformation to avoid log(0)

    # Scale the data (z-score normalization)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log_transformed)

    return data_scaled
