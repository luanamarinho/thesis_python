import numpy as np

def sampled_ind_matrix(metadata , nbr_samples = 2000, seed = 1234):
    """
    Generates row indices to downsample raw gene expression data, based on its metadata.
    The metadata is grouped by attributes 'PatientNumber', 'TumorType', 'CellType'. The indices of each subgroup are sampled, so that
    the vector's size is given by the product of group's size proportion in the original data set and the desired total number of samples in
    the downsampled data set. The goal is to keep the groups' proportion in the new, reduced data.

    Parameters:
    - metadata: Metadata.
    - nbr_samples: Number of rows of the downsampled data set.
    - seed: Integer to ensure reproducibility.

    Returns:
    - List of sampled row indices of the original data set.
    """
    # Calculate the proportion of observations for each unique combination
    metadata_subset = metadata[['PatientNumber', 'TumorType', 'CellType']]
    group_proportions = metadata_subset.groupby(['PatientNumber', 'TumorType', 'CellType']).size() / len(metadata_subset)
    
    # Calculate the number of rows to sample from each unique combination
    sampled_rows_per_group = (group_proportions * nbr_samples).astype(int).clip(lower=1)

    def sample_rows(group, count):
        return group.sample(n=count, replace=False).index.tolist()
    
    # Apply the sampling function to each group and convert the result to a list of indices
    np.random.seed(seed)
    sampled_indices = metadata_subset.groupby(['PatientNumber', 'TumorType', 'CellType']).apply(lambda x: sample_rows(x, count=sampled_rows_per_group[x.name])).tolist()

    # Flatten the list of lists and limit it to the desired number of indices
    sampled_indices = [index for sublist in sampled_indices for index in sublist][:nbr_samples]

    return sampled_indices
