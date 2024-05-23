import logging
from utils.affinities_cache import compute_affinities
from utils.run_opentsne import run_openTSNE_with_combinations
from utils.parameters_combination import generate_combinations
from joblib import dump, load
import gzip
import numpy as np
import time
import os


mount_path = '/mnt/batch/tasks/fsmounts/data'

# Configure logging
logging.basicConfig(filename=os.path.join(mount_path,'log/script.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load data
        data_file_path = os.path.join(mount_path,'data_preprocessed_40000_10HVG')
        data_file = gzip.GzipFile(data_file_path, "r"); expr_data_preprocessed = np.load(data_file)
        #np.random.seed(1234)
        #ind_to_sample = np.random.choice(expr_data_preprocessed.shape[0], size=200, replace=False)
        #ind_to_sample = np.load(os.path.join(mount_path,'ind_to_sample_200.npy'))
        #expr_data_preprocessed_sampled = expr_data_preprocessed #[ind_to_sample]

        # Theta [0.0, 0.25, 0.5, 0.75, 1.0]
        #combination = [(5, 4, 0.1, 0.8, 0.75)]
        #combination = [(5, 4, 0.1, 0.8, 0), (5, 4, 0.1, 0.8, 0.25), (5, 4, 0.1, 0.8, 0.75), (5, 4, 0.1, 0.8, 1)]
        perplexity = 25
        combination = generate_combinations(perplexity)
        combination_BH = [comb for comb in combination if comb[-1] != 0]
        #ind = slice(88,133)
        ind = int(len(combination_BH)/4)
        combination_BH_run = combination_BH[:ind]
        
        start_time= time.time()
        affinity_cache = compute_affinities(X=expr_data_preprocessed,
                                            perplexity_values=[perplexity])
        runtime_affinity = time.time() - start_time
        pipeline = run_openTSNE_with_combinations(combination_BH_run, X = expr_data_preprocessed, affinity_cache = affinity_cache, verbose=True)
        
        # Save results
        output = (pipeline, runtime_affinity)
        #combination_str = '_'.join(map(str, combination[0]))
        result_file_path = os.path.join(mount_path,'output/pipeline_multiples_perp25_0-45.joblib')
        dump(output, result_file_path)

        logging.info("Script executed successfully.")

    except Exception as e:
        # Log the error
        logging.error("An error occurred: %s", str(e), exc_info=True)  # exc_info=True logs the traceback

if __name__ == "__main__":
    main()
