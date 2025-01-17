import os
import time
import argparse
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
from functions.random_search import random_search_slim  # Assuming this is the non-parallel version
from slim_gsgp_lib_torch.datasets.data_loader import *
import pickle
from tqdm import tqdm

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Settings
pop_size = 100
n_iter = 200
n_iter_rs = 100
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
    if not os.path.exists('params'):
        try:
            os.makedirs('params')
        except Exception as e:
            print(f"Failed to create 'params' directory: {e}")
            exit(1)
    
    # Check if the file already exists
    try:
        with open(f'params/{pattern}.pkl', 'rb') as f:
            pickle.load(f)
        print(f"File already exists: {pattern}.pkl")
        return
    except FileNotFoundError:
        print(f"Calculating: {pattern}")
    except Exception as e:
        print(f"Error while checking file existence: {e}")

    # Logging start time
    with open('time_log.txt', 'a') as f:
        f.write(f'{dataset_name}, {time.time()}\n')

    # Random Search
    try:
        results = random_search_slim(
            X, y, dataset_name, scale=scale,
            p_train=p_train, iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
            struct_mutation=struct_mutation, show_progress=True,
            x_o=xo, save=False, mut_xo=mut_xo
        )
        print("Random search completed successfully.")
    except Exception as e:
        print(f"Error in random_search_slim: {e}")

    try:
        with open(f'params/{pattern}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"File saved: {pattern}.pkl")
    except Exception as e:
        print(f"Failed to save file for {dataset_name}: {e}")
        raise    


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

    # Delete the time log file
    with open('time_log.txt', 'w') as f:
        f.write('Dataset,Time\n')

    # Define tasks for both scaled and unscaled processing
    tasks = [(loader, True, False, False, False) for loader in datasets] + [(loader, False, False, False, False) for loader in datasets]  # Scaled                   and   unscaled
    tasks += [(loader, True, True, False, False) for loader in datasets] + [(loader, True, False, True, False) for loader in datasets]    # Just structure mutation  and   just structure xo  
    tasks += [(loader, True, True, True, False) for loader in datasets] + [(loader, True, True, False, True) for loader in datasets]      # Structure mutation + xo  and   structure mutation + mutation xo
    tasks += [(loader, False, False, False, True) for loader in datasets] + [(loader, True, True, True, True) for loader in datasets]     # Just xo_mut              and   all
    
    # Shuffle the tasks
    random.shuffle(tasks)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_dataset, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
                try:
                    future.result()  # Raise any exceptions from the worker processes
                except Exception as e:
                    print(f"Error in processing: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error in processing: {e}")
    finally:
        print("Process finished")
        exit
