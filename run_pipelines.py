import logging
from utils.affinities_cache import compute_affinities
from utils.run_opentsne import run_openTSNE_with_combinations
from joblib import dump, load
import gzip
import numpy as np

# Configure logging
logging.basicConfig(filename='/mnt/cs-luana-marinho-gmail-com-100320037b715e57/script.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load data
        data_file_path = '/mnt/cs-luana-marinho-gmail-com-100320037b715e57/originaldata_HGV_preproc_gzip'
        data_file = gzip.GzipFile(data_file_path, "r"); expr_data_preprocessed = np.load(data_file)
        expr_data_preprocessed_sampled = expr_data_preprocessed[np.random.choice(expr_data_preprocessed.shape[0], size=20000, replace=False)]

        # Compute affinities
        #np.linspace(5, 90, num=18, dtype=int).tolist()
        combination = [(80, 4, 0.1, 0.8, 0.5)]
        affinity_cache = compute_affinities(X=expr_data_preprocessed_sampled,
                                            perplexity_values=[80])
        pipeline = run_openTSNE_with_combinations(combination, X = expr_data_preprocessed_sampled, affinity_cache = affinity_cache, verbose=True)
        
        # Save results
        combination_str = '_'.join(map(str, combination[0]))
        result_file_path = f'/mnt/cs-luana-marinho-gmail-com-100320037b715e57/pipeline_{combination_str}.joblib'
        dump(pipeline, result_file_path)

        logging.info("Script executed successfully.")

    except Exception as e:
        # Log the error
        logging.error("An error occurred: %s", str(e), exc_info=True)  # exc_info=True logs the traceback

if __name__ == "__main__":
    main()
