import os

# Limit threads for NumPy and other multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads to 1
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads to 1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Limit NumExpr threads to 1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS threads to 1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Limit macOS Accelerate threads to 1
os.environ["BLIS_NUM_THREADS"] = "1"  # Limit BLIS threads to 1

from concurrent.futures import ProcessPoolExecutor, as_completed
from functions.test_algorithms import *
from slim_gsgp_lib.datasets.data_loader import *
import pickle
from tqdm import tqdm

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Settings
pop_size = 100
n_iter = 200
n_samples = 50
p_train = 0.7

def process_dataset(args):
    dataset_loader, scale, struct_mutation, xo, mut_xo = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__.split('load_')[1]

    # Get the suffixes for the file name
    scale_suffix = 'scaled' if scale else 'unscaled'
    xo_suffix = 'xo' if xo else 'no_xo'
    gp_xo_suffix = 'mut_xo' if mut_xo else 'no_mut_xo'
    struct_mutation_suffix = 'struct_mutation' if struct_mutation else 'no_struct_mutation'

    # Get the parameters from the random search
    try:
        with open(f'params/{dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl', 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"Failed to load file for {dataset_name}: {e}")
        return
    
    # Define the dictionary to store the results
    # rmse_, mape_, mae_, rmse_compare, mape_compare, mae_compare, time_stats, train_fit, test_fit, size, representations
    metrics = ['rmse', 'mape', 'mae', 'rmse_compare', 'mape_compare', 'mae_compare', 'time', 'train_fit', 'test_fit', 'size', 'representations']
    results = {metric: {} for metric in metrics}

    for algorithm in results:
        # Test SLIM
        rm, mp, ma, rm_c, mp_c, ma_c, time, train, test, size, reps = test_slim(
            X=X, y=y, args_dict=results[algorithm], dataset_name=dataset_loader.__name__,
            ms_lower=0, ms_upper=1, n_elites=1,
            iterations=n_samples, struct_mutation=struct_mutation, scale=scale, algorithm=algorithm,
            verbose=0, p_train=p_train, show_progress=False,
        )

    if not os.path.exists('results'):
        os.makedirs('results')

    try:
        with open(f'results/{dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl', 'wb') as f:
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
    max_workers = 120 

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()  # Raise any exceptions from the worker processes
            except Exception as e:
                print(f"Error in processing: {e}")
