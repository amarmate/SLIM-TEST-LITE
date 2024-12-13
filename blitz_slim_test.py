import os
import time
import subprocess  # For running Git commands
import argparse
import random
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functions.test_algorithms import *
from slim_gsgp_lib.datasets.data_loader import *
from slim_gsgp_lib.utils.callbacks import EarlyStopping
from sklearn.model_selection import KFold

# Limit threads for NumPy and other multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

datasets = [globals()[i] for i in globals() if 'load' in i][2:]
datasets = datasets[:12] + datasets[13:]  # EXCLUDE PARKINSONS

# Settings
pop_size = 40
n_iter = 2000
n_iter_rs = 50
n_samples = 30
p_train = 0.7


def process_dataset(args):
    dataset_loader, algorithm, scale, struct_mutation, xo, mut_xo = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__.split('load_')[1]
    algorithm_suffix = algorithm.replace('*', '_MUL_').replace('+', '_SUM_')

    # Suffix for file naming
    scale_suffix = 'scaled' if scale else None
    xo_suffix = 'xo' if xo else None
    gp_xo_suffix = 'mutxo' if mut_xo else None
    struct_mutation_suffix = 'strucmut' if struct_mutation else None
    pattern = '_'.join([i for i in [dataset_name, algorithm_suffix, scale_suffix, xo_suffix, gp_xo_suffix, struct_mutation_suffix] if i])
    pattern += '_new'  # TEMPORARY

    # Random search
    try:
        with open(f'params/{pattern}.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"Random search results already exist: {pattern}.pkl")
    except FileNotFoundError:
        print(f"Performing random search for: {pattern}")
        try:
            results = random_search_slim_cv(
                X, y, dataset_name, scale=scale,
                runs=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
                struct_mutation=struct_mutation, algorithm=algorithm,
                x_o=xo, mut_xo=mut_xo, pattern=pattern,
            )
            with open(f'params/{pattern}.pkl', 'wb') as f:
                pickle.dump(results, f)
                # save_and_commit(f'params/{pattern}.pkl', results)
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
            
            rm, mp, ma, rm_c, mp_c, ma_c, time_stats, size, reps = test_slim(
                X=X, y=y, args_dict=params_clean, dataset_name=dataset_loader.__name__,
                n_samples=n_samples, n_elites=1, scale=scale,
                algorithm=algorithm, verbose=1, p_train=p_train, show_progress=True,
            )
            
            test_results['rmse'][algorithm] = rm
            test_results['mape'][algorithm] = mp
            test_results['mae'][algorithm] = ma
            test_results['rmse_compare'][algorithm] = rm_c
            test_results['mape_compare'][algorithm] = mp_c
            test_results['mae_compare'][algorithm] = ma_c
            test_results['time'][algorithm] = time_stats
            test_results['size'][algorithm] = size
            test_results['representations'][algorithm] = reps

        with open(results_path, 'wb') as f:
            pickle.dump(test_results, f)
        print(f"Test results saved: {results_path}")
        # save_and_commit(results_path, test_results)
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process datasets with parallel workers.")
    parser.add_argument("--max-workers", type=int, default=96, help="Number of maximum workers for parallel processing.")
    args = parser.parse_args()
    if args.max_workers == -1:
        args.max_workers = os.cpu_count() - 1
    elif args.max_workers > os.cpu_count() - 1:
        print(f"Warning: Number of workers ({args.max_workers}) exceeds CPU count ({os.cpu_count()}).")
        args.max_workers = min(args.max_workers, os.cpu_count()-1)
    
    # Ensure domains exist 
    if not os.path.exists('params'):
        os.makedirs('params')
    if not os.path.exists('results/slim'):
        os.makedirs('results/slim')

    algorithms = ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
    # tasks = [(loader, True, False, False, False) for loader in datasets] + [(loader, False, False, False, False) for loader in datasets]
    # tasks += [(loader, True, True, False, False) for loader in datasets] + [(loader, True, False, True, False) for loader in datasets]
    # tasks += [(loader, True, True, True, False) for loader in datasets] + [(loader, True, True, False, True) for loader in datasets]
    # tasks += [(loader, False, False, False, True) for loader in datasets] + [(loader, True, True, True, True) for loader in datasets]
    
            # DATA  ,    ALGO  ,SCALE,STRUCT, XO,  MUT_XO
    tasks = [(loader, algorithm, True, True, False, False) for loader in datasets for algorithm in algorithms]
    tasks += [(loader, algorithm, True, False, False, False) for loader in datasets for algorithm in algorithms]
    # random.shuffle(tasks)

    tasks = tasks[:2]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
    
    



def random_search_slim_cv(X, y, dataset, pattern, scale=False,
                       runs=50, pop_size=100, n_iter=100, algorithm=None,
                       struct_mutation=False, x_o=False, mut_xo=False, 
                       test_elite=False, timeout=200):
    
    params = {
    'p_inflate': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7],
    'max_depth': [10,11,12,13],
    'init_depth': [4,5,6,7],
    'prob_const': [0.05, 0.1, 0.15, 0.2, 0.3],
    'tournament_size': [2, 3],
    'decay_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    'p_struct': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'depth_distribution': ['exp', 'uniform', 'norm'],
    
    # ----------------------- OTHER ---------------------------
    'p_xo': [0.1, 0.2, 0.3, 0.4, 0.5] if x_o==True or mut_xo==True else [0,0],  # If x_o or mut_xo is False, will be 0
    'p_struct_xo': (
        [0.25, 0.35, 0.5, 0.6, 0.7, 0.8] if x_o and mut_xo
        else [1, 1] if not mut_xo and x_o
        else [0, 0]
    ),
    }
    
    results_slim = {}
    # early_stopping = EarlyStopping(patience=500) if test_elite else None
    timeouts = 0
    if algorithm is None:
        algorithms = ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
    else:
        algorithms = [algorithm]
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
    for algo in algorithms:
        results = {}
        seed_ = random.randint(0, 10000)
            
            # if test_elite:
            #     dict_test_elite = {
            #         'X_test': X_test,
            #         'y_test': y_test,
            #         'callbacks': [early_stopping], 
            #     }

        for i in tqdm(range(runs)):
            # Randomly select parameters
            hyperparams = {
                'p_inflate': np.random.choice(params['p_inflate']),
                'max_depth': int(np.random.choice(params['max_depth'])),
                'init_depth': int(np.random.choice(params['init_depth'])),
                'tournament_size': int(np.random.choice(params['tournament_size'])),
                'prob_const': np.random.choice(params['prob_const']),
                'p_xo': np.random.choice(params['p_xo']),
                'p_struct_xo': np.random.choice(params['p_struct_xo']),
                'struct_mutation': struct_mutation,
                'decay_rate': np.random.choice(params['decay_rate']),
                'p_struct': np.random.choice(params['p_struct']),
                'depth_distribution': np.random.choice(params['depth_distribution']),
                'n_iter': n_iter,   
                'pop_size': pop_size,
            }

            # Ensure consistency between init_depth and max_depth
            if hyperparams['init_depth'] + 6 > hyperparams['max_depth']:
                hyperparams['max_depth'] = hyperparams['init_depth'] + 6

            
            scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                if scale:
                    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                    X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
                    X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
                    y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
                    y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            
                try:
                    slim_ = slim(X_train=X_train, y_train=y_train, dataset_name=dataset,
                                 slim_version=algo, **hyperparams, timeout=timeout, test_elite=False,
                                 verbose=0,
                                 )
                    
                    predictions = slim_.predict(X_test)
                    scores.append(rmse(y_test, predictions))
                    iteration = slim_.iteration

                    if iteration != n_iter and not slim_.early_stop:
                        timeouts += 1
                        print(f"Timeout {timeouts} ({iteration}) - Iteration {i} - Pattern: {pattern} - Seed: {seed_}")
                        print('Params:', hyperparams)
                        
                except Exception as e:
                    print(f"Exception occurred in random search: {str(e)}")
                    print(f"Iteration {i} - Pattern: {pattern} - Seed: {seed_}")
                    print('Params:', hyperparams)
                    continue
            
            rmse_score = np.mean(scores)
                        
            results[rmse_score] = hyperparams

        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
        # Get the best hyperparameters
        best_hyperparameters = list(results.values())[0]
        results_slim[algo] = best_hyperparameters
        print(f'Best RMSE for dataset {dataset} and algorithm {algo}: {list(results.keys())[0]}')

    return results_slim



def save_and_commit(filepath, data):
    """
    Saves data to a file and commits the change to GitHub.
    """
    # Save the file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"File saved: {filepath}")

    # Commit the change to GitHub
    try:
        # Stage the file
        subprocess.run(['git', 'add', filepath], check=True)
        # Commit with a message
        subprocess.run(['git', 'commit', '-m', f"Updated {os.path.basename(filepath)}"], check=True)
        # Push to the remote repository
        subprocess.run(['git', 'push'], check=True)
        print(f"File committed and pushed to GitHub: {filepath}")
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")