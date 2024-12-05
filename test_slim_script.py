import os
import time
import argparse
import random
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functions.test_algorithms import *
from functions.random_search import random_search_slim  # Assuming this is the non-parallel version
from slim_gsgp_lib.datasets.data_loader import *

# Limit threads for NumPy and other multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Settings
pop_size = 100
n_iter = 250
n_iter_rs = 100
n_samples = 50
p_train = 0.7

def process_dataset(args):
    dataset_loader, scale, struct_mutation, xo, mut_xo = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__.split('load_')[1]

    # Suffix for file naming
    scale_suffix = 'scaled' if scale else None
    xo_suffix = 'xo' if xo else None
    gp_xo_suffix = 'mut_xo' if mut_xo else None
    struct_mutation_suffix = 'struct_mutation' if struct_mutation else None
    pattern = '_'.join([i for i in [dataset_name, scale_suffix, xo_suffix, gp_xo_suffix, struct_mutation_suffix] if i])

    # Random search
    try:
        with open(f'params/{pattern}.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"Random search results already exist: {pattern}.pkl")
    except FileNotFoundError:
        print(f"Performing random search for: {pattern}")
        try:
            results = random_search_slim(
                X, y, dataset_name, scale=scale,
                p_train=p_train, iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
                struct_mutation=struct_mutation, show_progress=True,
                x_o=xo, save=False, mut_xo=mut_xo
            )
            with open(f'params/{pattern}.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"Random search completed and saved: {pattern}.pkl")
        except Exception as e:
            print(f"Error during random search: {e}")
            return

    # Load parameters for testing
    try:
        with open(f'params/{pattern}.pkl', 'rb') as f:
            params = pickle.load(f)
    except Exception as e:
        print(f"Failed to load parameters for {dataset_name}: {e}")
        return

    # Testing phase
    try:
        results_path = f'results/slim/{pattern}.pkl'
        if os.path.exists(results_path):
            print(f"Test results already exist: {results_path}")
            return

        metrics = ['rmse', 'mape', 'mae', 'rmse_compare', 'mape_compare', 'mae_compare', 'time', 'train_fit', 'test_fit', 'size', 'representations']
        test_results = {metric: {} for metric in metrics}

        for algorithm in tqdm(params.keys(), desc=f"Testing {pattern}"):
            params_clean = {k: (v.item() if isinstance(v, (np.float64, np.int64)) else v) for k, v in params[algorithm].items()}
            
            rm, mp, ma, rm_c, mp_c, ma_c, time_stats, train, test, size, reps = test_slim(
                X=X, y=y, args_dict=params_clean, dataset_name=dataset_loader.__name__,
                n_iter=n_iter, pop_size=pop_size, n_elites=1, iterations=n_samples, scale=scale,
                algorithm=algorithm, verbose=0, p_train=p_train, show_progress=False,
            )
            
            test_results['rmse'][algorithm] = rm
            test_results['mape'][algorithm] = mp
            test_results['mae'][algorithm] = ma
            test_results['rmse_compare'][algorithm] = rm_c
            test_results['mape_compare'][algorithm] = mp_c
            test_results['mae_compare'][algorithm] = ma_c
            test_results['time'][algorithm] = time_stats
            test_results['train_fit'][algorithm] = train
            test_results['test_fit'][algorithm] = test
            test_results['size'][algorithm] = size
            test_results['representations'][algorithm] = reps

        with open(results_path, 'wb') as f:
            pickle.dump(test_results, f)
        print(f"Test results saved: {results_path}")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process datasets with parallel workers.")
    parser.add_argument("--max-workers", type=int, default=96, help="Number of maximum workers for parallel processing.")
    args = parser.parse_args()
    
    # Ensure domains exist 
    if not os.path.exists('params'):
        os.makedirs('params')
    if not os.path.exists('results/slim'):
        os.makedirs('results/slim')

    tasks = [(loader, True, False, False, False) for loader in datasets] + [(loader, False, False, False, False) for loader in datasets]
    tasks += [(loader, True, True, False, False) for loader in datasets] + [(loader, True, False, True, False) for loader in datasets]
    tasks += [(loader, True, True, True, False) for loader in datasets] + [(loader, True, True, False, True) for loader in datasets]
    tasks += [(loader, False, False, False, True) for loader in datasets] + [(loader, True, True, True, True) for loader in datasets]

    random.shuffle(tasks)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
