import numpy as np
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri

def preprocess_sparse_matrix(sparse_input, HVG_indices, scale=True):
    """
    Preprocesses a sparse matrix by cell normalizing, log-transforming, and scaling features.
    
    Parameters:
        sparse_input (scipy.sparse.csr_matrix): Input sparse matrix to be preprocessed.
        HVG_indices (pd.Series boolean): boolean indicator of highly-variable genes.
        scale (bool): If True, the log-normalized and sliced data is also z-score scaled.
    Returns:
        numpy.ndarray: Log-normalized subset with identified HVG. If scale is True, the subset is z-score scaled.
    """

    scran = importr('scran')
    scater = importr('scater')
    SingleCellExperiment = importr('SingleCellExperiment')
    Matrix = importr('Matrix')

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_matrix = Matrix.Matrix(sparse_input.toarray(), sparse=True)

    sce = SingleCellExperiment.SingleCellExperiment(assays=ro.ListVector({'counts': r_matrix}))
    clusters = scran.quickCluster(sce)
    sce = scran.computeSumFactors(sce, clusters=clusters)
    sce = scater.logNormCounts(sce)

    norm_log_counts = ro.r['assay'](sce, 'logcounts')
    data_log_transformed = np.array(norm_log_counts)
    data_log_transformed = data_log_transformed[:, HVG_indices]

    if not scale:
        return data_log_transformed
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log_transformed)

    return data_scaled