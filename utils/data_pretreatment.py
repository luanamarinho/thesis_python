import numpy as np
from sklearn.preprocessing import normalize, StandardScaler

def preprocess_sparse_matrix(data_sp_csr_HVG):
    """
    Preprocesses a sparse matrix by normalizing, log-transforming, and scaling its values.
    
    Parameters:
        data_sparse (scipy.sparse.csr_matrix): Input sparse matrix to be preprocessed.
        
    Returns:
        numpy.ndarray: Preprocessed dense matrix.
    """
    data_dense = data_sp_csr_HVG.toarray()
    data_normalized = normalize(data_dense, norm='l2', axis=1)

    # Log transform the data
    data_log_transformed = np.log1p(data_normalized)  # Apply log(1+x) transformation to avoid log(0)

    # Scale the data (z-score normalization)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log_transformed)

    return data_scaled
