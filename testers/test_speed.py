from functions.test_algorithms import test_slim
from slim_gsgp_lib.datasets.data_loader import *
import numpy as np
from multiprocessing import Pool, cpu_count

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Repeat datasets 10 times 
datasets = datasets * 10

args = {
    'p_inflate': 0.3,
    'max_depth': 21,
    'init_depth': 10,
    'prob_const': 0.1,
    'prob_replace': 0.01,
    'p_prune': 0.6,
    'p_xo': 0.2,
    'p_struct_xo': 0.7
}

def process_dataset(dataset):
    X, y = dataset()
    rm, ma, nrmse, r2, mae, std_rmse, time, train, test, size = test_slim(
        X=X, y=y, args_dict=args, dataset_name='test',
        ms_lower=0, ms_upper=1, n_elites=1,
        iterations=4, scale=True, algorithm='SLIM*SIG1',
        verbose=0, p_train=0.7, show_progress=True,
    )
    rm = np.array(rm)
    rm_min = 100 * rm.min()
    return dataset.__name__, rm_min

if __name__ == '__main__':
    # Use all available CPU cores
    num_processes = cpu_count()
    
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Map the process_dataset function to all datasets
        results = pool.map(process_dataset, datasets)
    
    # Convert results to a dictionary
    results_dict = dict(results)
    
    # Print results
    for dataset_name, rm_min in results_dict.items():
        print(dataset_name, rm_min)