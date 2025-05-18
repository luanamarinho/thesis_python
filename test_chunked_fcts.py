from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances as pd
from sklearn.manifold import MDS
import numpy as np
from joblib import load, dump
import os
from utils.compute_distances import compute_distance_matrix_chunked
from utils.compute_trust_stress_chunked import compute_stress_trust
from sklearn.manifold import trustworthiness

# Test distance matrix
def check_equiv_distance(data):
    """
    Check if the distance matrix is equivalent to the one computed with sklearn chunked distances.
    """
    
    dist_matrix_sklearn = pd(data)
    dist_matrix_custom_chunks = compute_distance_matrix_chunked(data, n_jobs=1)

    assert np.array_equal(dist_matrix_sklearn, dist_matrix_custom_chunks), "Distance matrices are not equal!"


data = load_iris().data
check_equiv_distance(data)

data = np.random.RandomState(42).rand(1000, 30)
check_equiv_distance(data)
fname = "../data/dist_X_toy.pkl"
if not os.path.exists(fname):
    print("Saving toy distance matrix...")
    dist_matrix  = compute_distance_matrix_chunked(data, n_jobs=1)
    dump(dist_matrix, fname)

data = np.random.rand(5000, 30)
check_equiv_distance(data)

input_sc_data_path = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "logNormalized_HVG_subset_5000_samples_scaled_True.pkl")
data = load(input_sc_data_path)
check_equiv_distance(data)

fname = "../data/distance_matrix_preprocessed_5000.pkl"
if not os.path.exists(fname):
    print("Computing distance matrix...")
    dist_matrix = compute_distance_matrix_chunked(data, n_jobs=1)
    dump(dist_matrix, fname)

# Test MDS
mds_class = MDS(n_components=2, normalized_stress='auto', random_state=42)
data = np.random.RandomState(42).rand(1000, 30)
MDS_map_data= mds_class.fit_transform(data)
MDS_map_data.shape
mds_class.stress_ # non-normalized stress 391108.95636056364

fname = "../data/data_mds_toy.pkl"
if not os.path.exists(fname):
    print("Computing MDS map...")
    dump(MDS_map_data, fname)

# Run run_compute_trustworthiness_win_momentum.py in terminal with flag --toy_data
output_trust = np.memmap("../data/output_memmap_trust_toy_0_2_momentum.dat", dtype='float64', shape=(1, 2), mode='r')
output_stress = np.memmap("../data/output_memmap_stress_toy_0_2_momentum.dat", dtype='float64', shape=(1,), mode='r')

distances = load("../data/dist_X_toy.pkl", mmap_mode='r')
sum_squared_distances = (distances.ravel() ** 2).sum()

stress = mds_class.stress_ #391108.95636056364
stress = np.sqrt(stress / (sum_squared_distances / 2))

np.allclose(stress, output_stress[0], rtol=1e-4) # True


# Testing dependencies separately
def test_compute_stress_trust(data, k=[30]):
    """
    Test the compute_stress_trust function.
    """
    distances = compute_distance_matrix_chunked(data, n_jobs=1)
    
    mds = MDS(n_components=2, normalized_stress='auto', random_state=42)
    MDS_map_data = mds.fit_transform(data)
    stress_sklearn = mds.stress_
    trust_sklearn = trustworthiness(distances, MDS_map_data, n_neighbors=k[0], metric='precomputed')
    
    trust, stress_custom, stress_numerator = compute_stress_trust(distances, MDS_map_data, k=k)
    sum_squared_distances = (distances.ravel() ** 2).sum()
    stress_calculated = np.sqrt(stress_numerator / (sum_squared_distances / 2))

    print(f"Stress from sklearn: {np.round(stress_sklearn, 3)}")
    print(f"Stress numerator: {np.round(stress_numerator, 3)}")
    print(f"Stress ratio custom: {np.round(stress_custom, 3)}")
    print(f"Stress ratio calculated as in sklearn: {np.round(stress_calculated, 3)}")
    print(f"Trustworthiness custom: {np.round(trust, 3)}")
    print(f"Trustworthiness sklearn: {np.round(trust_sklearn, 3)}")

    # Check if the calculated stress is close to the expected value
    assert np.allclose(stress_custom, stress_calculated, rtol=1e-6) or np.allclose(stress_custom, stress_calculated, rtol=1e-3), "Stress values do not match!"
    assert np.allclose(trust, trust_sklearn, rtol=1e-6) or np.allclose(trust, trust_sklearn, rtol=1e-3), "Trustworthiness values do not match!"

data = load_iris().data
test_compute_stress_trust(data, k=[30])

data = np.random.RandomState(42).rand(1000, 30)
test_compute_stress_trust(data, k=[30])

data = np.random.RandomState(123).rand(1500, 30)
test_compute_stress_trust(data, k=[30])

data = np.random.RandomState(1234).rand(1000, 30)
test_compute_stress_trust(data, k=[30])
