import argparse
import logging
from utils.affinities_cache import compute_affinities
from utils.run_opentsne_momentum import run_openTSNE_with_combinations
from utils.parameters_combination_granular import generate_combinations
from joblib import dump, load
import gzip
import numpy as np
import time
import os

# Configure logging
logging.basicConfig(
    filename='log/script.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def tsne_pipelines(lower_bound: int, upper_bound: int, sampled: bool = False):
    """
    Produces t-SNE BH embeddings for multiple pipelines, derived from the tuples of parameter combinations
    (theta, initial momentum, final momentum, early exaggeration), for a fixed perplexity value.

    Parameters:
    perplexity (int): t-SNE perplexity. Must be one of [5, 25, 45, 65, 90].
    lower_bound (int): Lower bound index for the set of tuples.
    upper_bound (int): Upper bound index for the set of tuples.
    sampled (bool): If True, the input data will be randomly downsampled to 100 observations. Default is False.

    Returns:
    tuple: (pipelines, runtime_affinity)
        pipelines: List of tuples (combo, embedding_df, runtime, KL_divergence, pipeline_index).
        runtime_affinity: Time taken to compute affinities.
    """
    
    #assert perplexity in {5, 25, 45, 65, 90}, f"Invalid value: {perplexity}. Allowed values are [5, 25, 45, 65, 90]."
    
    try:
        # Load data
        data_file_path = 'C:/Users/luana/Documents/data/data_preprocessed_5000_10HVG'#'/home/luana/workspace/data/data_preprocessed_40000_10HVG'
        with gzip.GzipFile(data_file_path, "r") as data_file:
            expr_data_preprocessed = np.load(data_file)
        
        if sampled:
            ind_to_sample = np.load('inst/ind_to_sample_200.npy')[:100]
            expr_data_preprocessed = expr_data_preprocessed[ind_to_sample]

        # Generate combinations
        combinations = load('output/parameter_combinations.joblib')
        combinations_BH = [comb for comb in combinations if comb[-1] >= 0.1]
        combinations_BH_run = combinations_BH[lower_bound:upper_bound]

        # Compute affinities
        #start_time = time.time()
        #affinity_cache = compute_affinities(X=expr_data_preprocessed, perplexity_values=[perplexity])
        #runtime_affinity = time.time() - start_time

        # Run t-SNE with combinations
        pipelines = run_openTSNE_with_combinations(combinations_BH_run, X=expr_data_preprocessed, verbose=True)

        # Save results
        output = (pipelines)
        result_file_path = os.path.join("output", f"pipeline_multiples_{lower_bound}-{upper_bound}_debug_new.joblib")
        dump(output, result_file_path)

        logging.info("Script executed successfully with lower_bound %d, upper_bound %d", lower_bound, upper_bound)

    except Exception as e:
        logging.error("An error occurred: %s", str(e), exc_info=True)  # exc_info=True logs the traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run t-SNE pipelines with specified parameters.")
    parser.add_argument("--lower_bound", type=int, required=True, help="Lower bound index for the set of tuples.")
    parser.add_argument("--upper_bound", type=int, required=True, help="Upper bound index for the set of tuples.")
    parser.add_argument("--sampled", action='store_true', help="If set, the input data will be randomly downsampled to 100 observations.")
    
    args = parser.parse_args()
    
    tsne_pipelines(lower_bound=args.lower_bound, upper_bound=args.upper_bound, sampled=args.sampled)
