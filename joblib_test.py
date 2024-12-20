import os
import subprocess  
import argparse
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from slim_gsgp_lib.algorithms.SLIM_GSGP.operators.simplifiers import simplify_individual
from functions.test_algorithms import *
from slim_gsgp_lib.datasets.data_loader import *
from slim_gsgp_lib.utils.callbacks import EarlyStopping_train
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Limit threads for NumPy and other multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

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
p_train = 0.85
n_trials = 40  # 40
n_samples = 30 # 3

cv = 5 # 4
seed = 40
timeout = 100

def skopt_slim_cv(X, y, dataset, 
                  algorithm, 
                  scale=True, 
                  timeout=100, 
                  n_trials=50, 
                  max_iter=2000,
                  cv=4,
                  struct_mutation=False, 
                  struct_xo=False, 
                  mut_xo=False,
                  seed=0):
    trial_results = []  # To store mean RMSE and node counts for each trial

    def objective(params):
        init_depth, max_depth, pop_size, p_struct, p_inflate, tournament_size, prob_const, decay_rate, depth_distribution, p_xo, p_struct_xo = params
        if max_depth < init_depth + 6:
            return 10000000

        hyperparams = {
            'p_inflate': p_inflate,
            'max_depth': int(max_depth),
            'init_depth': int(init_depth),
            'tournament_size': int(tournament_size),
            'prob_const': prob_const,
            'struct_mutation': struct_mutation,
            'decay_rate': decay_rate,
            'p_struct': p_struct,
            'depth_distribution': depth_distribution,
            'pop_size': int(pop_size),
            'n_iter': int(max_iter),
            'p_xo': p_xo,
            'p_struct_xo': p_struct_xo,
        }

        # Perform K-fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        nodes_count = []
        early_stopping = EarlyStopping_train(patience=int(8_000 / pop_size))

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
                    callbacks=[early_stopping],
                    seed=seed
                )

                slim_ = simplify_individual(slim_, y_train, X_train, threshold=0)
                predictions = slim_.predict(X_test)
                scores.append(rmse(y_test, predictions))
                nodes_count.append(slim_.nodes_count)
            except Exception as e:
                print(f"Exception: {e}")
                return float("inf")

        mean_rmse = np.mean(scores)
        mean_node_count = np.mean(nodes_count)

        # Store results for later processing
        trial_results.append((mean_rmse, mean_node_count, hyperparams))
        return mean_rmse

    # Define search space with parameter names
    space = [
        Integer(3, 12, name='init_depth'),
        Integer(9, 24, name='max_depth'),
        Integer(25, 200, name='pop_size', prior='uniform'),
        Real(0, 0.3, name='p_struct', prior='uniform'),
        Real(0, 0.7, name='p_inflate', prior='uniform'),
        Integer(2, 5, name='tournament_size'),
        Real(0.05, 0.3, name='prob_const', prior='uniform'),
        Real(0, 0.35, name='decay_rate', prior='uniform'),
        Categorical(['exp', 'uniform', 'norm'], name='depth_distribution'),
]

    if mut_xo and struct_xo:
        space.append(Real(0, 1, name='p_xo', prior='uniform'))
        space.append(Real(0, 1, name='p_struct_xo', prior='uniform'))
    
    elif struct_xo:
        space.append(Real(0, 1, name='p_xo', prior='uniform'))
        space.append(Real(1, 1, name='p_struct_xo', prior='uniform'))
    
    elif mut_xo:
        space.append(Real(1, 1, name='p_xo', prior='uniform'))
        space.append(Categorical([0], name='p_struct_xo'))

    else: 
        space.append(Categorical([0], name='p_xo'))
        space.append(Categorical([0], name='p_struct_xo'))


    result = gp_minimize(
        func=lambda params: objective(params),
        dimensions=space,
        n_calls=n_trials,
        random_state=seed,
    )

    # Post-processing to find the best parameters
    rmses, nodes, params_list = zip(*trial_results)
    rmses = np.array(rmses)
    nodes = np.array(nodes)

    # Standardize both metrics
    standardized_rmse = (rmses - rmses.mean()) / rmses.std()
    standardized_nodes = (nodes - nodes.mean()) / nodes.std()

    # Combine metrics and find the best parameters
    combined_metric = standardized_rmse + 0.5 * standardized_nodes
    best_index = np.argmin(combined_metric)
    best_params = params_list[best_index]

    # print(f"Best parameters: {best_params}")
    # print(f"Best RMSE: {rmses[best_index]}")
    # print(f"Best size: {nodes[best_index]}")

    return best_params

def process_dataset(dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo):
    dataset_id = dataset_dict[name]
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
        print(f"Random search results already exist: {name} - {pattern}.pkl")
    except FileNotFoundError:
        print(f"Performing random search for: {name} - {pattern}")
        try:
            results = skopt_slim_cv(
                X=X_train,
                y=y_train,
                dataset=name,
                algorithm=algorithm,
                scale=scale,
                timeout=timeout,
                n_trials=n_trials,
                max_iter=max_iter,
                cv=cv,
                struct_mutation=struct_mutation,
                struct_xo=xo,
                mut_xo=mut_xo,
                seed=seed
            )

            with open(f'params/{dataset_id}/{pattern}.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"Random search completed and saved: {pattern}.pkl")

        except Exception as e:
            print(f"Error during random search: {e}")
            return

    # Load parameters for testing
    try:
        with open(f'params/{dataset_id}/{pattern}.pkl', 'rb') as f:
            params = pickle.load(f)
    except Exception as e:
        print(f"Failed to load parameters for {name} - {pattern}: {e}")
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
            args_dict=params, dataset_name=name,
            n_samples=n_samples, n_elites=1        )

        # Store results in the dictionary
        test_results['rmse'] = rm
        test_results['mape'] = mp
        test_results['mae'] = ma
        test_results['rmse_compare'] = rm_c
        test_results['mape_compare'] = mp_c
        test_results['mae_compare'] = ma_c
        test_results['time'] = time_stats
        test_results['size'] = size
        test_results['representations'] = reps

        # Save test results to a file
        with open(results_path, 'wb') as f:
            pickle.dump(test_results, f)

        print(f"Testing completed and results saved: {results_path}")

    except Exception as e:
        print(f"Error during testing for {name} - {pattern}: {e}")

# Parallel execution
def main():
    parser = argparse.ArgumentParser(description="Run SLIM-GSGP experiments.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs (default: -1 for all available cores)")
    args = parser.parse_args()

    # Create a list of loaded split datasets
    data = [(dataset_loader(), dataset_loader.__name__.split('load_')[1]) for dataset_loader in datasets]
    data_split = [(train_test_split(X, y, p_test=1-p_train, shuffle=True, seed=seed), name) for (X, y), name in data]

    parallel_jobs = args.n_jobs
    print(f"Running experiments with {parallel_jobs} parallel jobs...")

    # Define dataset and algorithm combinations
    experiments = [
        (dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo)
        for dataset, name in data_split
        for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
        for scale in [True]
        for struct_mutation in [True, False]
        for xo in [False]
        for mut_xo in [False]
        for gp_xo in [False]
    ]

    print(f"Total number of experiments: {len(experiments)}")

    # Execute experiments in parallel
    Parallel(n_jobs=parallel_jobs)(
        delayed(process_dataset)(dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo)
        for dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo in experiments
    )

if __name__ == "__main__":
    main()
