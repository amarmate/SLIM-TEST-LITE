import os
import time

# Limit threads for NumPy and other multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads to 1
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads to 1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Limit NumExpr threads to 1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS threads to 1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Limit macOS Accelerate threads to 1
os.environ["BLIS_NUM_THREADS"] = "1"  # Limit BLIS threads to 1

from concurrent.futures import ProcessPoolExecutor, as_completed
from functions.test_algorithms import *
from functions.random_search import random_search_slim  # Assuming this is the non-parallel version
from slim_gsgp_lib.datasets.data_loader import *
import pickle
from tqdm import tqdm

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Settings
pop_size = 100
n_iter = 200
n_iter_rs = 100
p_train = 0.7

def process_dataset(args):
    scale_suffix = 'scaled' if scale else 'unscaled'
    xo_suffix = 'xo' if xo else 'no_xo'
    gp_xo_suffix = 'mut_xo' if mut_xo else 'no_mut_xo'
    struct_mutation_suffix = 'struct_mutation' if struct_mutation else 'no_struct_mutation'

    # Logging start time
    if not os.path.exists('time_log.txt'):
        with open('time_log.txt', 'w') as f:
            f.write('Dataset,Time\n')
    with open('time_log.txt', 'a') as f:
        f.write(f'{dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix},{time.time()}\n')
    
    if not os.path.exists('params'):
        os.makedirs('params')
    
    # Check if the there are already results for the dataset
    try:
        with open(f'params/{dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl', 'wb') as f:
            pickle.load(f)
        print(f"File already exists: {dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl")
        return
    except Exception as e:
        pass

    dataset_loader, scale, struct_mutation, xo, mut_xo = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__

    # Random Search
    results = random_search_slim(
        X, y, dataset_name, scale=scale,
        p_train=p_train, iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
        struct_mutation=struct_mutation, show_progress=True,
        x_o=xo, save=False, mut_xo=mut_xo
    )

    try:
        with open(f'params/{dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"File saved: {dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl")
    except Exception as e:
        print(f"Failed to save file for {dataset_name}: {e}")    


if __name__ == '__main__':
    # Define tasks for both scaled and unscaled processing
    tasks = [(loader, True, False, False, False) for loader in datasets] + [(loader, False, False, False, False) for loader in datasets]
    tasks += [(loader, True, True, False, False) for loader in datasets] + [(loader, True, False, True, False) for loader in datasets]
    tasks += [(loader, True, True, True, False) for loader in datasets] + [(loader, True, True, True, True) for loader in datasets]

    # Limit max workers to balance CPU and memory usage
    max_workers = 96 

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()  # Raise any exceptions from the worker processes
            except Exception as e:
                print(f"Error in processing: {e}")
