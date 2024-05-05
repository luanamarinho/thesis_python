from anndata import AnnData
import scanpy as sc

def slice_data_HVG(expr_data, perc_top_genes = 0.1):
    """
    Retrieves n_top highly variable genes and accordingly slices expression data matrix.

    Parameters:
    - expr_data: Input raw gene expression data matrix, where the rows are cells, and the columns are the features.
    - perc_top_genes: Float indicating the percentage of the top HVG genes to be retrieved.

    Returns:
    - A sliced expression data matrix with only the n_top HVG.
    """
    adata = AnnData(X = expr_data)
    sc.pp.highly_variable_genes(adata, n_top_genes=int(perc_top_genes * adata.shape[1]), flavor='seurat_v3')
    highly_variable_genes = adata.var['highly_variable']
    sliced_X = adata.X[:, highly_variable_genes]

    return sliced_X
