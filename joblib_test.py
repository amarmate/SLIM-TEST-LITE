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
import time 

from matplotlib import pyplot as plt

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
n_trials = 40    # 40  75
n_samples = 50   # 50

cv = 4           # 5
seed = 100        # 40
timeout = 45     # 45

iter_dict = {     # EarlyStop
    '30' : 2300,  # 551 
    '60' : 1900,  # 295
    '90' : 1600,  # 205
    '120': 1400,  # 158 
    '150': 1200,   # 129
    '175': 1000,   # 95
    '200': 800,   # 85
}

def skopt_slim_cv(X, y, dataset, 
                  algorithm, 
                  pattern,
                  scale=True, 
                  timeout=100, 
                  n_trials=50, 
                  max_iter=2000,
                  cv=4,
                  struct_mutation=False, 
                  struct_xo=False, 
                  mut_xo=False,
                  random_state=0, 
                  simplify=False
                  ):
    trial_results = []
    calls_count = 0

    def objective(params):
        nonlocal calls_count
        start = time.time()
        init_depth, max_depth, pop_size, p_struct, p_inflate, tournament_size, prob_const, decay_rate, depth_distribution, p_xo, p_struct_xo, simplify_threshold = params
        if max_depth < init_depth + 6:
            calls_count += 1
            return 100000
        
        if p_struct + p_inflate*2 > 50:
            calls_count += 1
            return 100000
        
        # n_iter = int(pop_size * 30)
        # n_iter = iter_dict[str(n_iter)]
        n_iter = 2000

        hyperparams = {
            'p_inflate': p_inflate / 25,
            'max_depth': int(max_depth),
            'init_depth': int(init_depth),
            'tournament_size': int(tournament_size),
            'prob_const': prob_const / 50,
            'struct_mutation': struct_mutation,
            'decay_rate': decay_rate / 50,
            'p_struct': p_struct / 50,
            'depth_distribution': depth_distribution,
            # 'pop_size': int(pop_size * 30),
            'pop_size': int(pop_size),
            'n_iter': int(n_iter),
            'p_xo': p_xo / 25,
            'p_struct_xo': p_struct_xo / 25,
            'initializer': 'simple',
        }

        # Perform K-fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        nodes_count = []
        # early_stopping = EarlyStopping_train(patience=int(12_000 / pop_size**0.9))   # 10_000
        early_stopping = EarlyStopping_train(patience=250)

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

                slim_ = simplify_individual(slim_, y_train, X_train, threshold=simplify_threshold/100) if simplify else slim_
                predictions = slim_.predict(X_test)
                scores.append(rmse(y_test, predictions))
                nodes_count.append(slim_.nodes_count)
            except Exception as e:
                print(f"Exception: {e}")
                return float("inf")

        mean_score = np.mean(scores)
        # std_score = np.std(scores)
        mean_node_count = np.mean(nodes_count)

        hyperparams['simplify_threshold'] = simplify_threshold / 100 if simplify else None
        trial_results.append((mean_score, mean_node_count, hyperparams))
        best_trial = min(trial_results, key=lambda x: x[0])

        print(f"Trial {calls_count + 1}/{n_trials} - Pattern: {dataset}-{pattern} - Time: {time.time() - start:.2f}s")
        print(f"RMSE: {mean_score:.4f} (Best: {best_trial[0]:.4f}) - Nodes: {mean_node_count:.2f} (Best: {best_trial[1]:.2f})")
        print(f"Parameters: {hyperparams}\n")
        calls_count += 1
        return mean_score

    # Define search space with parameter names
    space = [
        Integer(5, 10, name='init_depth', prior='uniform'),   # 3 - 10
        Integer(11, 23, name='max_depth'),                     # 9 - 22
        # Integer(1, 5, name='pop_size', prior='uniform'),     # * 30
        Categorical([100], name='pop_size'),
        Integer(0, int(35/2), name='p_struct', prior='uniform'),    # / 50
        Integer(0, int(70/4), name='p_inflate', prior='uniform'),   # / 25
        # Integer(2, 4, name='tournament_size'),
        Categorical([2], name='tournament_size'),
        Integer(0, int(30/2), name='prob_const', prior='uniform'),  # / 50
        # Integer(0, int(30/2), name='decay_rate', prior='uniform'),  # / 50
        Categorical([0], name='decay_rate'),
        # Categorical(['exp', 'uniform', 'norm'], name='depth_distribution'),
        Categorical(['diz'], name='depth_distribution'),
]

    # All to be divided by 25
    if mut_xo and struct_xo:
        space.append(Real(0, 25, name='p_xo', prior='uniform'))
        space.append(Real(0, 25, name='p_struct_xo', prior='uniform'))
    
    elif struct_xo:
        space.append(Real(0, 25, name='p_xo', prior='uniform'))
        space.append(Real(25, 25, name='p_struct_xo', prior='uniform'))
    
    elif mut_xo:
        space.append(Real(25, 25, name='p_xo', prior='uniform'))
        space.append(Categorical([0], name='p_struct_xo'))

    else: 
        space.append(Categorical([0], name='p_xo'))
        space.append(Categorical([0], name='p_struct_xo'))

    if simplify:
        space.append(Categorical([1], name='simplify_threshold'))
        # space.append(Integer(-1, 3, name='simplify_threshold'))

    else:
        space.append(Categorical([None], name='simplify_threshold'))

    gp_minimize(
        func=lambda params: objective(params),
        dimensions=space,
        n_calls=n_trials,
        random_state=random_state,
        verbose=False,
        noise=1e-2,    # Noise level, check for better convergence
        n_random_starts=20,
    )

    # Post-processing to find the best parameters
    scores, nodes, params_list = zip(*trial_results)
    scores = np.array(scores)
    nodes = np.array(nodes)

    # Find the pareto front
    pareto_front = np.zeros(len(scores), dtype=bool)
    for i, (score, node) in enumerate(zip(scores, nodes)):
        pareto_front[i] = np.sum(np.logical_and(scores < score, nodes < node)) == 0

    # Filtert the pareto front
    scores = scores[pareto_front]
    nodes = nodes[pareto_front]
    params_list = np.array(params_list)[pareto_front]

    # Standardize both metrics
    standardized_rmse = (scores - scores.mean()) / scores.std()
    standardized_nodes = (nodes - nodes.mean()) / nodes.std()

    # Combine metrics and find the best parameters
    combined_metric = standardized_rmse + 0.5 * standardized_nodes
    best_index = np.argmin(combined_metric)
    best_params = params_list[best_index]

    # print(f"Best parameters: {best_params}")
    # print(f"Best RMSE: {rmses[best_index]}")
    # print(f"Best size: {nodes[best_index]}")

    return best_params


def process_dataset(dataset, name, algorithm, 
                    scale, struct_mutation, xo, 
                    mut_xo, gp_xo, simplify=None,
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
    simplify_threshold_suffix = 'sp' if simplify else None
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
                simplify=simplify,
                pattern=pattern,
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

        # early_stopping = EarlyStopping_train(patience=int(12_000/params['pop_size']))

        # Remove the simplify_threshold from the parameters
        params_test = params.copy()
        simplify_threshold = params_test.pop('simplify_threshold')
            
        rm, mp, ma, rm_c, mp_c, ma_c, time_stats, size, reps = test_slim(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            args_dict=params_test, dataset_name=name, verbose=0,
            n_samples=n_samples, n_elites=1, simplify_threshold=simplify_threshold,
            # callbacks=[early_stopping],
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
        (dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify)
        for dataset, name in data_split
        for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
        for scale in [True]
        for struct_mutation in [False]
        for xo in [False]
        for mut_xo in [False]
        for gp_xo in [False]
        for simplify in [False]
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
        delayed(process_dataset)(dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify, random_state)
        for dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify, random_state in experiments
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