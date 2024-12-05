import os
import argparse
import time
import random

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
n_samples = 30
p_train = 0.7

def process_dataset(args):
    dataset_loader, scale, struct_mutation, xo, mut_xo = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__.split('load_')[1]
    
    # Get the suffixes for the file name
    scale_suffix = 'scaled' if scale else None
    xo_suffix = 'xo' if xo else None
    gp_xo_suffix = 'mut_xo' if mut_xo else None
    struct_mutation_suffix = 'struct_mutation' if struct_mutation else None
    pattern = '_'.join([i for i in [dataset_name, scale_suffix, xo_suffix, gp_xo_suffix, struct_mutation_suffix] if i])
    
    # Check if the directory exists
    if not os.path.exists('results/slim'):
        try:
            os.makedirs('results/slim')
        except Exception as e:
            print(f"Failed to create 'results/slim' directory: {e}")
            exit(1)
            
    # Check if there are already results for the dataset
    try:
        with open(f'results/slim/{pattern}.pkl', 'rb') as f:
            pickle.load(f)
        print(f"File already exists: {pattern}.pkl")
        return
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error while checking file existence: {e}")


    # Log when the process starts 
    start_time = time.time()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    if not os.path.exists('time_logs.txt'):
        with open('time_logs.txt', 'w') as f:
            f.write('Dataset,Time\n')
    with open('time_logs.txt', 'a') as f:
        f.write(f'{pattern},{start_time}\n')

    # Get the parameters from the random search
    try:
        with open(f'params/{pattern}.pkl', 'rb') as f:
            params = pickle.load(f)
    except Exception as e:
        print(f"Failed to load file for {dataset_name}: {e}")
        return
        
    # Define the dictionary to store the results
    metrics = ['rmse', 'mape', 'mae', 'rmse_compare', 'mape_compare', 'mae_compare', 'time', 'train_fit', 'test_fit', 'size', 'representations']
    results = {metric: {} for metric in metrics}

    try:
        for algorithm in tqdm(params.keys(), desc=f"Testing {pattern}"):
            # Clean the dictionary
            params_clean = {k: (v.item() if isinstance(v, (np.float64, np.int64)) else v) for k, v in params[algorithm].items()}
            
            # Test SLIM
            rm, mp, ma, rm_c, mp_c, ma_c, time_stats, train, test, size, reps = test_slim(
                X=X, y=y, args_dict=params_clean, dataset_name=dataset_loader.__name__,
                n_iter=n_iter, pop_size=pop_size, n_elites=1, iterations=n_samples, scale=scale, 
                algorithm=algorithm, verbose=0, p_train=p_train, show_progress=False,
            )
            
            # Store the results in a compact manner
            results['rmse'][algorithm] = rm
            results['mape'][algorithm] = mp
            results['mae'][algorithm] = ma
            results['rmse_compare'][algorithm] = rm_c
            results['mape_compare'][algorithm] = mp_c
            results['mae_compare'][algorithm] = ma_c
            results['time'][algorithm] = time_stats
            results['train_fit'][algorithm] = train
            results['test_fit'][algorithm] = test
            results['size'][algorithm] = size
            results['representations'][algorithm] = reps
    except Exception as e:
        print(f"Error in testing SLIM: {e}")
        return

    try:
        with open(f'results/slim/{pattern}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"File saved: {pattern}.pkl")
    except Exception as e:
        print(f"Failed to save file for {dataset_name}: {e}")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process datasets with parallel workers.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=96, 
        help="Number of maximum workers for parallel processing."
    )
    args = parser.parse_args()
    max_workers = args.max_workers

    # Define tasks for both scaled and unscaled processing
    tasks = [(loader, True, False, False, False) for loader in datasets] + [(loader, False, False, False, False) for loader in datasets]  # Scaled                   and   unscaled
    tasks += [(loader, True, True, False, False) for loader in datasets] + [(loader, True, False, True, False) for loader in datasets]    # Just structure mutation  and   just structure xo  
    tasks += [(loader, True, True, True, False) for loader in datasets] + [(loader, True, True, False, True) for loader in datasets]      # Structure mutation + xo  and   structure mutation + mutation xo
    tasks += [(loader, False, False, False, True) for loader in datasets] + [(loader, True, True, True, True) for loader in datasets]     # Just xo_mut              and   all
    
    random.shuffle(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()  # Raise any exceptions from the worker processes
            except Exception as e:
                print(f"Error in processing: {e}")
