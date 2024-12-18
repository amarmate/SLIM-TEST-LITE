import os
import time
import subprocess  
import argparse
import random
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from slim_gsgp_lib.algorithms.SLIM_GSGP.operators.simplifiers import simplify_individual
from functions.test_algorithms import *
from slim_gsgp_lib.datasets.data_loader import *
from slim_gsgp_lib.utils.callbacks import EarlyStopping_train
from sklearn.model_selection import KFold
import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# Limit threads for NumPy and other multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["OPTUNA_THREAD_COUNT"] = "1"

datasets = [globals()[i] for i in globals() if 'load' in i][2:]
datasets = datasets[:12] + datasets[13:]  # EXCLUDE PARKINSONS
dataset_dict = {}
for i, dataset in enumerate(datasets):
    X,y = dataset()
    name = dataset.__name__.split('load_')[1]
    id = 'DA' + str(i).zfill(2)
    dataset_dict[name] = id 

# Settings
max_iter = 2000  # 2000
p_train = 0.8
n_trials = 40  # 40
n_samples = 30 # 3

cv = 4 # 4
seed = 40
timeout = 100


def optuna_slim_cv(X, y, dataset, 
                   algorithm, 
                   scale=True, 
                   timeout=100, 
                   n_trials=50, 
                   max_iter=2000,
                   cv=4,
                   struct_mutation=False, 
                   xo=False, 
                   struct_xo=False, 
                   mut_xo=False):
    
    def objective(trial):
        init_depth = trial.suggest_int('init_depth', 3, 12)
        max_depth = trial.suggest_int('max_depth', init_depth + 6, 24)
        pop_size = trial.suggest_int('pop_size', 25, 200, step=25)
        p_struct = trial.suggest_float('p_struct', 0, 0.3, step=0.05)

        if xo: 
            p_xo = trial.suggest_float('p_xo', 0, 0.5, step=0.05)
            if struct_xo and mut_xo:
                p_struct_xo = trial.suggest_float('p_struct_xo', 0, 1, step=0.1)
            elif struct_xo:
                p_struct_xo = 1
            elif mut_xo:
                p_struct_xo = 0
        else: 
            p_xo, p_struct_xo = 0, 0

        hyperparams = {
            'p_inflate': trial.suggest_float('p_inflate', 0, 0.7, step=0.05),
            'max_depth': max_depth,
            'init_depth': init_depth,
            'tournament_size': trial.suggest_int('tournament_size', 2, 5),
            'prob_const': trial.suggest_float('prob_const', 0.05, 0.3, step=0.025),
            'struct_mutation': struct_mutation,
            'decay_rate': trial.suggest_float('decay_rate', 0, 0.35, step=0.05),
            'p_struct': p_struct,
            'depth_distribution': trial.suggest_categorical('depth_distribution', ['exp', 'uniform', 'norm']),
            'pop_size': pop_size,
            'n_iter': max_iter,
            'p_xo' : p_xo,
            'p_struct_xo': p_struct_xo,
        }

        # Perform K-fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        nodes_count = []
        EarlyStopping = EarlyStopping_train(patience=int(8_000/pop_size))

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
                slim_ = slim(
                    X_train=X_train,
                    y_train=y_train,
                    dataset_name=dataset,
                    slim_version=algorithm,
                    **hyperparams,
                    timeout=timeout,
                    test_elite=False,
                    verbose=0,
                    callbacks=[EarlyStopping]
                )

                slim_ = simplify_individual(slim_, y_train, X_train, threshold=0)
                predictions = slim_.predict(X_test)
                scores.append(rmse(y_test, predictions))
                nodes_count.append(slim_.nodes_count)
            except Exception as e:
                print(f"Exception: {e}")
                return float("inf")

        return np.mean(scores), np.mean(nodes_count)

    # Create an Optuna study
    study = optuna.create_study(directions=['minimize', 'minimize'], study_name='SLIM_GSGP')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # Get the best results
    pareto_trials = study.best_trials
    rmse_values = np.array([trial.values[0] for trial in pareto_trials])
    size_values = np.array([trial.values[1] for trial in pareto_trials])

    # Normalize RMSE and size using min-max scaling
    rmse_min, rmse_max = rmse_values.min(), rmse_values.max()
    size_min, size_max = size_values.min(), size_values.max()
    rmse_normalized = (rmse_values - rmse_min) / (rmse_max - rmse_min)
    size_normalized = (size_values - size_min) / (size_max - size_min)

    # Compute combined scores
    combined_scores = rmse_normalized + 0.5 * size_normalized
    best_index = np.argmin(combined_scores)
    best_trial = pareto_trials[best_index]

    # print("Best RMSE:", best_trial.values[0])
    # print("Best Size:", best_trial.values[1])
    # print("Best Hyperparameters:", best_trial.params)

    return best_trial.params


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
        subprocess.run(['git', 'add', filepath], check=True)
        subprocess.run(['git', 'commit', '-m', f"Updated {os.path.basename(filepath)}"], check=True)
        subprocess.run(['git', 'push'], check=True)
        print(f"File committed and pushed to GitHub: {filepath}")

    except subprocess.CalledProcessError as e:

        print(f"Git operation failed: {e}")


def process_dataset(args):
    dataset, dataset_name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo = args
    dataset_id = dataset_dict[dataset_name]
    algorithm_suffix = algorithm.replace('*', 'MUL_').replace('+', 'SUM_').replace('SLIM', '')
    X_train, X_test, y_train, y_test = dataset

    # Suffix for file naming
    scale_suffix = 'sc' if scale else None
    xo_suffix = 'xo' if xo else None
    gp_xo_suffix = 'gx' if gp_xo else None
    mut_xo_suffix = 'mx' if mut_xo else None
    struct_mutation_suffix = 'sm' if struct_mutation else None
    pattern = algorithm_suffix + '_' + ''.join([i for i in [scale_suffix, xo_suffix, gp_xo_suffix, mut_xo_suffix, struct_mutation_suffix] if i])

    # Ensure the domain exists
    if not os.path.exists(f'params/{dataset_id}'):
        os.makedirs(f'params/{dataset_id}')
    if not os.path.exists(f'results/slim/{dataset_id}'):
        os.makedirs(f'results/slim/{dataset_id}')

    # Random search
    try:
        with open(f'params/{dataset_id}/{pattern}.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"Random search results already exist: {dataset_name} - {pattern}.pkl")
    except FileNotFoundError:
        print(f"Performing random search for: {dataset_name} - {pattern}")
        try:
            results = optuna_slim_cv(
                X=X_train, y=y_train, dataset=dataset_name, algorithm=algorithm, scale=scale, timeout=timeout,
                n_trials=n_trials, max_iter=max_iter, struct_mutation=struct_mutation, xo=xo, 
                struct_xo=gp_xo, mut_xo=mut_xo, cv=cv, 
            )

            with open(f'params/{dataset_id}/{pattern}.pkl', 'wb') as f:
                pickle.dump(results, f)
                # save_and_commit(f'params/{dataset_id}/{pattern}.pkl', results)
            print(f"Random search completed and saved: {pattern}.pkl")

        except Exception as e:
            print(f"Error during random search: {e}")
            return

    # Load parameters for testing
    try:
        with open(f'params/{dataset_id}/{pattern}.pkl', 'rb') as f:
            params = pickle.load(f)
    except Exception as e:
        print(f"Failed to load parameters for {dataset_name} - {pattern}: {e}")
        return

    # Testing phase
    try:
        results_path = f'results/slim/{dataset_id}/{pattern}.pkl'
        if os.path.exists(results_path):
            print(f"Test results already exist: {results_path}")
            return

        metrics = ['rmse', 'mape', 'mae', 'rmse_compare', 'mape_compare', 'mae_compare', 'time', 'train_fit', 'test_fit', 'size', 'representations']
        test_results = {metric: {} for metric in metrics}

        EarlyStopping = EarlyStopping_train(patience=int(8_000/params['pop_size']))
            
        rm, mp, ma, rm_c, mp_c, ma_c, time_stats, size, reps = test_slim(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            args_dict=params, dataset_name=dataset_name,
            n_samples=n_samples, n_elites=1, scale=scale,
            algorithm=algorithm, verbose=0, p_train=p_train, show_progress=True, timeout=timeout,
            callbacks=[EarlyStopping],
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

    # Create a list of loaded split datasets
    data = [(dataset_loader(), dataset_loader.__name__.split('load_')[1]) for dataset_loader in datasets]
    data_split = [(train_test_split(X, y, p_test=1-p_train, shuffle=True, seed=seed), name) for (X, y), name in data]

    algorithms = ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]

            # DATA  ,    ALGO  ,SCALE,STRUCT, XO,  MUT_XO, GP_XO
    tasks = [(dataset, name, algorithm, True, True, False, False, False) for (dataset, name) in data_split for algorithm in algorithms]
    tasks += [(dataset, name, algorithm, True, False, False, False, False) for (dataset, name) in data_split for algorithm in algorithms]
    # random.shuffle(tasks)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_dataset, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing tasks"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
            
    # Aggregate the dictionary of params and results, as they were saved in separate files
    params_dict = {}
    results_dict = {}

    for dataset in datasets:
        try:
            dataset_name = dataset.__name__.split('load_')[1]
            dataset_id = dataset_dict[dataset_name]

            # The file will have this pattern: algorithm_scxo.pkl. 
            # We need to ensure only that for some settings (ex.: scxo) all the algorithms are present
            avalaible_settings = []
            for file in os.listdir(f'params/{dataset_id}'):
                # Get the different settings available 
                settings = file.split('_')[2].split('.')[0]
                if settings not in avalaible_settings:
                    avalaible_settings.append(settings)
            
            for settings in avalaible_settings:
                dict_params = {}
                dict_results = {}
                for suffix in ['MUL_ABS', 'MUL_SIG1', 'MUL_SIG2', 'SUM_ABS', 'SUM_SIG1', 'SUM_SIG2']:
                    try:
                        params = pickle.load(open(f'params/{dataset_id}/{suffix}_{settings}.pkl', 'rb'))
                        dict_params.update(params)
                        results = pickle.load(open(f'results/slim/{dataset_id}/{suffix}_{settings}.pkl', 'rb'))
                        for k, v in results.items():
                            if k not in dict_results:
                                dict_results[k] = {}
                            dict_results[k].update(v)

                        # Delete the loaded pickle files 
                        os.remove(f'params/{dataset_id}/{suffix}_{settings}.pkl')
                        os.remove(f'results/slim/{dataset_id}/{suffix}_{settings}.pkl')
                    except Exception as e:
                        print(f"Error in loading or deleting files: {e}")
                        continue
                
                print(dict_params)
                print(dict_results)

                # Dump the results
                pickle.dump(dict_params, open(f'params/{dataset_id}/{settings}.pkl', 'wb'))
                pickle.dump(dict_results, open(f'results/slim/{dataset_id}/{settings}.pkl', 'wb')) 

        except Exception as e:
            print(f"Error in processing dataset: {e}")
            continue       