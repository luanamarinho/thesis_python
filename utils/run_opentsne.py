import time
import pandas as pd
import openTSNE

def run_openTSNE_with_combinations(combinations, X, affinity_cache, initialization='random', n_jobs=1, 
                                   negative_gradient_method='BH', random_state=1234, 
                                   n_iter=750, verbose=False, dof=1):
    """
    Run openTSNE with a list of parameter combinations.

    Parameters:
    - combinations: List of parameter combinations.
    - X: Input data matrix.
    - affinity_cache: Dictionary containing pre-computed affinities.
    - Other parameters as in the previous function.

    Returns:
    - List of tuples, each containing the combination and the resulting embedding.
    """
    results = []
    i = 0 

    for combo in combinations:
        perplexity, early_exagg, initial_momentum, final_momentum, theta = combo

        # Get pre-computed affinities from cache
        affinities = affinity_cache[perplexity]

        # Create TSNE object with the provided parameters
        tsne = openTSNE.TSNE(
            perplexity=perplexity,
            early_exaggeration=early_exagg,
            initialization=initialization,
            n_jobs=n_jobs,
            negative_gradient_method=negative_gradient_method,
            theta=theta,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state,
            dof=dof
        )

        start_time = time.time()

        # Fit TSNE with cached affinities
        embedding = tsne.fit(affinities=affinities)
        
        runtime = time.time() - start_time
        
        pipeline = 'pipeline_' + str(i)
        embedding_df = pd.DataFrame({
            'TSNE_1_' + pipeline: embedding[:, 0],
            'TSNE_2_' + pipeline: embedding[:, 1],
        })

        KL_divergence = embedding.kl_divergence
        
        # Append the results
        results.append((combo, embedding_df, runtime, KL_divergence, pipeline))

    return results