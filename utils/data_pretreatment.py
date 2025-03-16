import numpy as np
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri

def preprocess_sparse_matrix(data_sp_csr_HVG):
    """
    Preprocesses a sparse matrix by cell normalizing, log-transforming, and scaling features.
    
    Parameters:
        data_sparse (scipy.sparse.csr_matrix): Input sparse matrix to be preprocessed.        
    Returns:
        numpy.ndarray: Preprocessed dense matrix.
    """

    scran = importr('scran')
    scater = importr('scater')
    SingleCellExperiment = importr('SingleCellExperiment')
    Matrix = importr('Matrix')

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_matrix = Matrix.Matrix(data_sp_csr_HVG.toarray(), sparse=True)

    sce = SingleCellExperiment.SingleCellExperiment(assays=ro.ListVector({'counts': r_matrix}))
    clusters = scran.quickCluster(sce)
    sce = scran.computeSumFactors(sce, clusters=clusters)
    sce = scater.logNormCounts(sce)

    norm_log_counts = ro.r['assay'](sce, 'logcounts')
    data_log_transformed = np.array(norm_log_counts)

    # Scale the data (z-score normalization)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log_transformed)

    return data_scaled