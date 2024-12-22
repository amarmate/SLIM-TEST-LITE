import os
import argparse
import pickle
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
p_train = 0.8    # 0.85
n_trials = 40    # 40
n_samples = 50   # 50

cv = 6           # 6
seed = 0         # 40
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
                  random_state=0, 
                  simplify_threshold=None):
    trial_results = []

    def objective(params):
        init_depth, max_depth, pop_size, p_struct, p_inflate, tournament_size, prob_const, decay_rate, depth_distribution, p_xo, p_struct_xo = params
        if max_depth < init_depth + 6:
            return 100000
        
        if p_struct + p_inflate > 1:
            return 100000

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
        early_stopping = EarlyStopping_train(patience=int(7_000 / pop_size**0.9))   # 10_000

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

                slim_ = simplify_individual(slim_, y_train, X_train, threshold=simplify_threshold) if simplify_threshold else slim_
                predictions = slim_.predict(X_test)
                scores.append(rmse(y_test, predictions))
                nodes_count.append(slim_.nodes_count)
            except Exception as e:
                print(f"Exception: {e}")
                return float("inf")

        mean_rmse = np.mean(scores)
        std_rmse = np.std(scores)
        mean_node_count = np.mean(nodes_count)

        # Store results for later processing
        trial_results.append((mean_rmse, mean_node_count, hyperparams))
        return mean_rmse / (1+std_rmse)

    # Define search space with parameter names
    space = [
        Integer(3, 12, name='init_depth'),
        Integer(9, 24, name='max_depth'),
        Integer(25, 200, name='pop_size', prior='uniform'),
        Real(0, 0.5, name='p_struct', prior='uniform'),
        Real(0, 0.7, name='p_inflate', prior='uniform'),
        Integer(2, 5, name='tournament_size'),
        Real(0, 0.3, name='prob_const', prior='uniform'),
        Real(0, 0.4, name='decay_rate', prior='uniform'),
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


    gp_minimize(
        func=lambda params: objective(params),
        dimensions=space,
        n_calls=n_trials,
        random_state=random_state,
        verbose=True
    )

    # Post-processing to find the best parameters
    rmses, nodes, params_list = zip(*trial_results)
    rmses = np.array(rmses)
    nodes = np.array(nodes)

    # Standardize both metrics
    standardized_rmse = (rmses - rmses.mean()) / rmses.std()
    standardized_nodes = (nodes - nodes.mean()) / nodes.std()

    # Combine metrics and find the best parameters
    combined_metric = standardized_rmse + 0.7 * standardized_nodes
    best_index = np.argmin(combined_metric)
    best_params = params_list[best_index]

    # print(f"Best parameters: {best_params}")
    # print(f"Best RMSE: {rmses[best_index]}")
    # print(f"Best size: {nodes[best_index]}")

    return best_params

def process_dataset(dataset, name, algorithm, 
                    scale, struct_mutation, xo, 
                    mut_xo, gp_xo, simplify_threshold=None,
                    random_state=0):
    
    dataset_id = dataset_dict[name]
    algorithm_suffix = algorithm.replace('*', 'MUL_').replace('+', 'SUM_').replace('SLIM', '')
    X_train, X_test, y_train, y_test = dataset

    # Suffix for file naming
    scale_suffix = 'sc' if scale else None
    xo_suffix = 'xo' if xo else None
    gp_xo_suffix = 'gx' if gp_xo else None
    mut_xo_suffix = 'mx' if mut_xo else None
    struct_mutation_suffix = 'sm' if struct_mutation else None
    simplify_threshold_suffix = 'sp' if simplify_threshold else None
    pattern = algorithm_suffix + '_' + ''.join([i for i in [scale_suffix, xo_suffix, gp_xo_suffix, 
                                                            mut_xo_suffix, struct_mutation_suffix,
                                                            simplify_threshold_suffix] if i])

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
                random_state=random_state,
                simplify_threshold=simplify_threshold, 
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

        early_stopping = EarlyStopping_train(patience=int(7_000/params['pop_size']**0.9))    # Added
            
        rm, mp, ma, rm_c, mp_c, ma_c, time_stats, size, reps = test_slim(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            args_dict=params, dataset_name=name,
            n_samples=n_samples, n_elites=1, simplify_threshold=simplify_threshold,
            callbacks=[early_stopping],
        )

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


def divide_tasks(tasks, num_chunks):
    """Divide tasks into specified number of chunks."""
    chunk_size = (len(tasks) + num_chunks - 1) // num_chunks  # Ceiling division
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]


# Parallel execution
def main():
    parser = argparse.ArgumentParser(description="Run SLIM-GSGP experiments.")
    parser.add_argument('--max-workers', type=int, default=-1, help="Number of parallel workers.")
    parser.add_argument('--divide', type=str, default="1:0", 
                        help="Format: NUM_CHUNKS:CHUNK_INDEX (e.g., 3:1 for the second chunk of 3).")
    args = parser.parse_args()

    # Create a list of loaded split datasets
    data = [(dataset_loader(), dataset_loader.__name__.split('load_')[1]) for dataset_loader in datasets]
    data_split = [(train_test_split(X, y, p_test=1-p_train, shuffle=True, seed=seed), name) for (X, y), name in data]

    parallel_jobs = args.max_workers if args.max_workers > 0 else os.cpu_count()
    num_chunks = int(args.divide.split(':')[0])
    chunk_index = int(args.divide.split(':')[1])
    
    print(f"Running experiments with {parallel_jobs} parallel jobs...")

    # Define dataset and algorithm combinations
    experiments = [
        (dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify_threshold)
        for dataset, name in data_split
        for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
        for scale in [True, False]
        for struct_mutation in [True, False]
        for xo in [False]
        for mut_xo in [False]
        for gp_xo in [False]
        for simplify_threshold in [None]
    ]

    # Add to each experiment a random_state 
    for i,_ in enumerate(experiments):
        experiments[i] += (seed+i,)

    # Divide tasks into chunks
    chunks = divide_tasks(experiments, num_chunks)
    experiments = chunks[chunk_index]
    
    print(f"Total number of experiments: {len(experiments)}")

    # Execute experiments in parallel
    Parallel(n_jobs=parallel_jobs)(
        delayed(process_dataset)(dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify_threshold, random_state)
        for dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify_threshold, random_state in experiments
    )

if __name__ == "__main__":
    main()

    # # Aggregate the dictionary of params and results, as they were saved in separate files
    # params_dict = {}
    # results_dict = {}

    # for dataset in datasets:
    #     try:
    #         dataset_name = dataset.__name__.split('load_')[1]
    #         dataset_id = dataset_dict[dataset_name]

    #         # The file will have this pattern: algorithm_scxo.pkl. 
    #         # We need to ensure only that for some settings (ex.: scxo) all the algorithms are present
    #         avalaible_settings = []
    #         for file in os.listdir(f'results/slim/{dataset_id}'):
    #             # Get the different settings available 
    #             if len(file.split('_')) < 3:
    #                 continue
    #             settings = file.split('_')[2].split('.')[0]
    #             if settings not in avalaible_settings:
    #                 avalaible_settings.append(settings)
            
    #         for settings in avalaible_settings:
    #             dict_params = {}
    #             dict_results = {}
    #             for suffix in ['MUL_ABS', 'MUL_SIG1', 'MUL_SIG2', 'SUM_ABS', 'SUM_SIG1', 'SUM_SIG2']:
    #                 try:
    #                     # Parameters
    #                     with open(f'params/{dataset_id}/{suffix}_{settings}.pkl', 'rb') as f:
    #                         params = pickle.load(f)
    #                     params = {suffix : params}
    #                     dict_params.update(params)
    #                     os.remove(f'params/{dataset_id}/{suffix}_{settings}.pkl')
    #                 except Exception as e:
    #                     print(f"Error in parameters {dataset_id} - {settings}: {e}")

    #                 try:
    #                     # Results
    #                     with open(f'results/slim/{dataset_id}/{suffix}_{settings}.pkl', 'rb') as f:
    #                         results = pickle.load(f)
    #                     for k, v in results.items():
    #                         if k not in dict_results:
    #                             dict_results[k] = {}
    #                         v = {suffix : v}
    #                         dict_results[k].update(v)
    #                     os.remove(f'results/slim/{dataset_id}/{suffix}_{settings}.pkl')

    #                 except Exception as e:
    #                     print(f"Error in results {dataset_id} - {settings}: {e}")
    #                     continue
                
    #             # Dump the results
    #             pickle.dump(dict_params, open(f'params/{dataset_id}/{settings}.pkl', 'wb'))
    #             pickle.dump(dict_results, open(f'results/slim/{dataset_id}/{settings}.pkl', 'wb')) 

    #     except Exception as e:
    #         print(f"Error in processing dataset: {e}")
    #         continue       