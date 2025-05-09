import os
import argparse
import pickle
from joblib import Parallel, delayed
from slim_gsgp_lib_np.algorithms.SLIM_GSGP.operators.simplifiers import simplify_individual
from functions.test_algorithms import *
from slim_gsgp_lib_np.datasets.data_loader import *
from slim_gsgp_lib_np.utils.callbacks import EarlyStopping_train
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import time 
from matplotlib import pyplot as plt

# Limit threads  
# for NumPy and other multi-threaded libraries
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
max_iter = 2000 
p_train = 0.8    
n_trials = 50 
n_samples = 50    
cv = 4          
seed = 200        
timeout = 100   

def skopt_slim_cv(X, y, dataset, 
                  algorithm, 
                  pattern,
                  scale=True, 
                  timeout=100, 
                  n_trials=50, 
                  max_iter=2000,
                  cv=4,
                  struct_mutation=False, 
                  no_structure=False
                  ):
    trial_results = []
    calls_count = 0

    def objective(params):
        nonlocal calls_count
        start = time.time()
        init_depth, max_depth, pop_size, p_struct, p_inflate, tournament_size, prob_const, decay_rate, depth_distribution, p_xo, p_struct_xo, simplify_threshold, initializer = params
        if max_depth < init_depth + 6:
            calls_count += 1
            return 100000
        
        if p_struct + p_inflate*2 > 50:
            calls_count += 1
            return 100000

        n_iter = 2000
        hyperparams = {
            'p_inflate': p_inflate / 30,
            'max_depth': int(max_depth),
            'init_depth': int(init_depth),
            'tournament_size': int(tournament_size),
            'prob_const': prob_const / 50,
            'decay_rate': decay_rate / 20,
            'p_struct': p_struct / 50,
            'depth_distribution': depth_distribution,
            'pop_size': int(pop_size),
            'n_iter': int(n_iter),
            'p_xo': p_xo / 25,
            'p_struct_xo': p_struct_xo / 25,
            'initializer': initializer,
        }

        # Perform K-fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        nodes_count = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if scale:
                scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                X_train = scaler_x.fit_transform(X_train)
                X_test = scaler_x.transform(X_test)
                y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
                y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

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
        Integer(3, 8, name='init_depth', prior='uniform'),
        Integer(10, 20, name='max_depth'),                   
        Categorical([100], name='pop_size'),
        Integer(0.02, int(20), name='p_struct', prior='uniform') if struct_mutation else Categorical([0], name='p_struct'),
        Integer(0, int(20), name='p_inflate', prior='uniform'),   # / 30, max = 0.6(6)
        Categorical([2, 3, 4], name='tournament_size'),
        Integer(0, int(15), name='prob_const', prior='uniform'),  # / 50, max = 0.3
        Integer(0, int(15), name='decay_rate', prior='uniform'),  # / 20, max = 0.75
        Categorical(['exp', 'uniform', 'norm'], name='depth_distribution'),
]

    if no_structure:
        space.append(Categorical(['simple'], name='initializer'))
    
    else:
        space.append(Categorical(['rhh'], name='initializer'))

    gp_minimize(
        func=lambda params: objective(params),
        dimensions=space,
        n_calls=n_trials,
        random_state=seed,
        verbose=False,
        noise=2e-2,    # Noise level, check for better convergence
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
    scores_pf = scores[pareto_front]
    nodes_pf = nodes[pareto_front]
    params_list = np.array(params_list)[pareto_front]

    # Standardize both metrics
    standardized_rmse = (scores_pf - scores_pf.mean()) / scores_pf.std()
    standardized_nodes = (nodes_pf - nodes_pf.mean()) / nodes_pf.std()

    # Combine metrics and find the best parameters
    combined_metric = standardized_rmse + 0.4 * standardized_nodes
    best_index = np.argmin(combined_metric)
    best_params = params_list[best_index]

    # print(f"Best parameters: {best_params}")
    # print(f"Best RMSE: {rmses[best_index]}")
    # print(f"Best size: {nodes[best_index]}")

    return best_params, scores, nodes, space


def process_dataset(dataset, name, algorithm, 
                    scale, struct_mutation, xo, 
                    mut_xo, gp_xo, simplify=None,
                    no_structure=False
                    ):
    
    dataset_id = dataset_dict[name]
    algorithm_suffix = algorithm.replace('*', 'MUL_').replace('+', 'SUM_').replace('SLIM', '')
    X_train, X_test, y_train, y_test = dataset

    # Suffix for file naming
    scale_suffix = 'sc' if scale else 'un'
    struct_mutation_suffix = 'sm' if struct_mutation else None
    simplify_threshold_suffix = 'sp' if simplify else None
    no_structure_suffix = 'ns' if no_structure else None
    pattern = algorithm_suffix + '_' + ''.join([i for i in [scale_suffix, xo_suffix, gp_xo_suffix, 
                                                            mut_xo_suffix, struct_mutation_suffix,
                                                            simplify_threshold_suffix, no_structure_suffix] if i])

    # Ensure the domain exists
    if not os.path.exists(f'params/{dataset_id}'):
        os.makedirs(f'params/{dataset_id}', exist_ok=True)
    if not os.path.exists(f'results/slim/{dataset_id}'):
        os.makedirs(f'results/slim/{dataset_id}', exist_ok=True)

    # Random search
    try:
        with open(f'params/{dataset_id}/{pattern}.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"Random search results already exist: {name} - {pattern}.pkl")
    except FileNotFoundError:
        print(f"Performing random search for: {name} - {pattern}")
        try:
            best_params, scores, nodes, space = skopt_slim_cv(
                X=X_train,
                y=y_train,
                dataset=name,
                algorithm=algorithm,
                scale=scale,
                timeout=timeout,
                n_trials=n_trials,
                max_iter=max_iter,
                cv=cv,
                simplify=simplify,
                no_structure=no_structure,
                pattern=pattern,
            )

            with open(f'params/{dataset_id}/{pattern}.pkl', 'wb') as f:
                # Compile the results in a dictionary
                results = {
                    'best_params': best_params,
                    'scores': scores,
                    'nodes': nodes,
                    'space': space,
                }
                pickle.dump(results, f)
            print(f"Random search completed and saved: {pattern}.pkl")

        except Exception as e:
            print(f"Error during random search: {e}")
            return

    # Load parameters for testing
    try:
        with open(f'params/{dataset_id}/{pattern}.pkl', 'rb') as f:
            params = pickle.load(f)['best_params']
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

        # Remove the simplify_threshold from the parameters
        params_test = params.copy()
            
        rm, mp, ma, rm_c, mp_c, ma_c, time_stats, size, reps = test_slim(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            args_dict=params_test, dataset_name=name, verbose=0, scale=scale,
            n_samples=n_samples, n_elites=1,
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
    parser = argparse.ArgumentParser(description="Run SLIM-GSGP trials.")
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
    
    print(f"Running trials with {parallel_jobs} parallel jobs...")

    # Define dataset and algorithm combinations
    trials = [
        (dataset, name, algorithm, scale, struct_mutation, xo, mut_xo, gp_xo, simplify, no_structure)
        for dataset, name in data_split
        for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
        for scale in [True, False]
        for struct_mutation in [False]
        for no_structure in [True]
    ]

    # Add to each experiment a random_state 
    for i,_ in enumerate(trials):
        trials[i] += (seed+i,)
        
    # Divide tasks into chunks
    chunks = divide_tasks(trials, num_chunks)
    trials = chunks[chunk_index]
    
    print(f"Total number of trials: {len(trials)}")

    # Execute trials in parallel
    Parallel(n_jobs=parallel_jobs)(
        delayed(process_dataset)(dataset, name, algorithm, scale, struct_mutation, no_structure, random_state)
        for dataset, name, algorithm, scale, struct_mutation, no_structure, random_state in trials
    )

if __name__ == "__main__":
    main()





















# OLD CODE ---------------------------------
    # # All to be divided by 25
    # if mut_xo and struct_xo:
    #     space.append(Real(0, 25, name='p_xo', prior='uniform'))
    #     space.append(Real(0, 25, name='p_struct_xo', prior='uniform'))
    
    # elif struct_xo:
    #     space.append(Real(0, 25, name='p_xo', prior='uniform'))
    #     space.append(Real(25, 25, name='p_struct_xo', prior='uniform'))
    
    # elif mut_xo:
    #     space.append(Real(25, 25, name='p_xo', prior='uniform'))
    #     space.append(Categorical([0], name='p_struct_xo'))

    # else: 
    #     space.append(Categorical([0], name='p_xo'))
    #     space.append(Categorical([0], name='p_struct_xo'))