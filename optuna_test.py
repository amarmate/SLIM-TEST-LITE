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
import optuna

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

def optuna_slim_cv(X, y, dataset, pattern, algorithm, scale=False, timeout=200, n_trials=50, struct_mutation=False):
    def objective(trial):
        # Define hyperparameter search space
        hyperparams = {
            'p_inflate': trial.suggest_float('p_inflate', 0, 0.7, step=0.05),
            'max_depth': trial.suggest_int('max_depth', 9, 24),
            'init_depth': trial.suggest_int('init_depth', 3, 12),
            'tournament_size': trial.suggest_int('tournament_size', 2, 6),
            'prob_const': trial.suggest_float('prob_const', 0.05, 0.3, step=0.025),
            'struct_mutation': struct_mutation,
            'decay_rate': trial.suggest_float('decay_rate', 0.05, 0.35, step=0.05),
            'p_struct': trial.suggest_float('p_struct', 0, 0.4, step=0.05),
            'depth_distribution': trial.suggest_categorical('depth_distribution', ['exp', 'uniform', 'norm']),
            'pop_size': trial.suggest_int('pop_size', 40, 200, step=10),
        }

        # Ensure consistency between init_depth and max_depth
        if hyperparams['init_depth'] + 6 > hyperparams['max_depth']:
            hyperparams['max_depth'] = hyperparams['init_depth'] + 6

        hyperparams['n_iter'] = 125000 / hyperparams['pop_size']**{1.2}

        # Perform K-fold cross-validation
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Scale the data if needed
            if scale:
                scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
                X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
                y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
                y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

            try:
                slim_ = slim(
                    X_train=X_train,
                    y_train=y_train,
                    dataset_name=dataset,
                    slim_version=algorithm,
                    **hyperparams,
                    timeout=timeout,
                    test_elite=False,
                    verbose=0,
                )

                predictions = slim_.predict(X_test)
                scores.append(rmse(y_test, predictions))
            except Exception as e:
                print(f"Exception: {e}")
                return float("inf")  # Penalize failed trials

        # Return the average RMSE across all folds
        return np.mean(scores)

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials) 

    # Get the best results
    best_hyperparams = study.best_params
    best_score = study.best_value

    print("Best RMSE:", best_score)
    print("Best hyperparameters:", best_hyperparams)

    return best_hyperparams



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
                x_o=xo, mut_xo=mut_xo, pattern=pattern, timeout=30,
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
                algorithm=algorithm, verbose=0, p_train=p_train, show_progress=True, timeout=40,
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



# ------------------------------ MAIN ------------------------------
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
    # tasks += [(loader, algorithm, True, False, False, False) for loader in datasets for algorithm in algorithms]
    # random.shuffle(tasks)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
    
    



