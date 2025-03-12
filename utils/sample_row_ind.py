import numpy as np

def sampled_ind_matrix(metadata, nbr_samples = 2000, seed = 42, col_names = ['CellType']):
    """
    Generates row indices to downsample raw gene expression data, based on its metadata.
    The metadata is grouped by attributes 'PatientNumber', 'TumorType', 'CellType'. The indices of each subgroup are sampled, so that
    the vector's size is given by the product of group's size proportion in the original data set and the desired total number of samples in
    the downsampled data set. The goal is to keep the groups' proportion in the new, reduced data.

    Parameters:
    - metadata: Metadata.
    - nbr_samples: Maximum number of rows of the downsampled data set.
    - seed: Integer to ensure reproducibility.
    - col_names: A list with the column name(s) to group

    Returns:
    - List of sampled row indices of the original data set.
    """
    metadata_subset = metadata[col_names]
    group_proportions = metadata_subset.groupby(col_names).size() / len(metadata_subset)
    
    sampled_rows_per_group = np.ceil((group_proportions * nbr_samples)).astype(int)

    def sample_rows(group, count, set_seed):
        return group.sample(n=count, replace=False, random_state = set_seed).index.tolist()
    
    sampled_indices = metadata_subset.groupby(col_names).apply(lambda x: sample_rows(x, count=sampled_rows_per_group[x.name], set_seed=seed)).tolist()

    sampled_indices = [index for sublist in sampled_indices for index in sublist][:nbr_samples]

    return sampled_indices
