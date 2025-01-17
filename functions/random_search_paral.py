from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os
import pickle
from slim_gsgp_lib_torch.main_slim import slim
from slim_gsgp_lib_torch.utils.utils import train_test_split
from slim_gsgp_lib_torch.evaluators.fitness_functions import rmse
import torch
from sklearn.preprocessing import MinMaxScaler

def random_search_slim_parallel(X, y, dataset, scale=False, p_train=0.7,
                                iterations=50, pop_size=100, n_iter=100,
                                struct_mutation=False, show_progress=True, 
                                x_o=False, gp_xo=False, save=True, identifier=None):
    """
    Perform a random search for the best hyperparameters for the SLIM algorithm in parallel.

    Parameters
    ----------
    Same as the original function.

    Returns
    -------
    results_slim: dict
        A dictionary containing the best hyperparameters for each SLIM algorithm.
    """
    
    algorithms = ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]
    
    params = {
        'p_inflate': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7],
        'max_depth': [19, 20, 21, 22, 23, 24],
        'init_depth': [5, 6, 7, 8, 10],
        'prob_const': [0.05, 0.1, 0.15, 0.2, 0.3],
        'tournament_size': [2, 3],
        'ms_lower': [0, 0, 0, 0.05, 0.1],
        'ms_upper': [1, 1, 1, 1, 0.8, 0.6, 0.4],
        'p_prune': [0.1, 0.2, 0.3, 0.4, 0.5] if struct_mutation else [0, 0],
        'p_xo': [0.1, 0.2, 0.3, 0.4, 0.5] if x_o else [0, 0],
        'p_struct_xo': (
            [0.25, 0.35, 0.5, 0.6, 0.7, 0.8] if x_o and gp_xo
            else [1, 1] if not gp_xo and x_o
            else [0, 0]
        ),
        'prob_replace': [0, 0.01, 0.015, 0.02] if struct_mutation else [0, 0],
    }

    # Split dataset once
    X_train, X_test, y_train, y_test = train_test_split(X=X,y=y,p_test=1-p_train)

    if scale:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
        y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1 ,1)).reshape(-1), dtype=torch.float32)
        y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1 ,1)).reshape(-1), dtype=torch.float32)

    def evaluate_single_iteration(args):
        seed , algorithm = args
        np.random.seed(seed)
        
        p_inflate = np.random.choice(params['p_inflate'])
        max_depth = int(np.random.choice(params['max_depth']))
        init_depth = int(np.random.choice(params['init_depth']))
        tournament_size = int(np.random.choice(params['tournament_size']))
        prob_const = np.random.choice(params['prob_const'])
        prob_replace = np.random.choice(params['prob_replace'])
        p_prune = np.random.choice(params['p_prune'])
        p_xo = np.random.choice(params['p_xo'])
        p_struct_xo = np.random.choice(params['p_struct_xo'])
        ms_lower = int(np.random.choice(params['ms_lower']))
        ms_upper = int(np.random.choice(params['ms_upper']))

        if init_depth +6 > max_depth:
            max_depth = init_depth +6

        slim_ = slim(X_train=X_train,y_train=y_train,dataset_name='dataset_1',
                     X_test=X_test,y_test=y_test,
                     slim_version=algorithm,pop_size=pop_size,n_iter=n_iter,
                     ms_lower=ms_lower ,ms_upper=ms_upper ,p_inflate=p_inflate,max_depth=max_depth ,init_depth=init_depth,
                     seed=seed ,prob_const=prob_const ,n_elites=1 ,log_level=0 ,verbose=0,
                     struct_mutation=struct_mutation ,prob_replace=prob_replace ,p_prune=p_prune ,
                     p_xo=p_xo ,p_struct_xo=p_struct_xo,tournament_size=tournament_size)

        predictions_slim = slim_.predict(X_test)
        
        rmse_score = float(rmse(y_true=y_test,y_pred=predictions_slim))
        
        return rmse_score,{
            'algorithm': algorithm,
            'p_inflate': p_inflate,
            'max_depth': max_depth,
            'init_depth': init_depth,
            'prob_const': prob_const,
            'prob_replace': prob_replace,
            'p_prune': p_prune,
            'p_xo': p_xo,
            'p_struct_xo': p_struct_xo,
            'struct_mutation': struct_mutation,
            'tournament_size': tournament_size,
            'ms_lower': ms_lower,
            'ms_upper': ms_upper,
         }

    # Prepare all combinations of seeds and algorithms for parallel execution
    tasks = [(seed ,algorithm) for seed in range(iterations) for algorithm in algorithms]

    # Use ProcessPoolExecutor to parallelize the iterations across all tasks
    with ProcessPoolExecutor() as executor:
         results = list(tqdm(executor.map(evaluate_single_iteration,tasks),total=len(tasks),disable=not show_progress))

    # Sort results and get the best parameters for each algorithm
    results_slim = {}
    
    for algorithm in algorithms:
         sorted_results_for_algorithm = sorted([result for result in results if result[1]['algorithm'] == algorithm],key=lambda x: x[0])
         results_slim[algorithm] = sorted_results_for_algorithm[0][1]

    # Save the results if needed
    if save:
         output_dir = os.path.join(os.getcwd(),"best_params")
         os.makedirs(output_dir ,exist_ok=True)
         output_file = f"best_slim_{dataset}_{pop_size}_{n_iter}_{scale}.pkl"
         
         if identifier:
             output_file = f"best_slim_{dataset}_{pop_size}_{n_iter}_{scale}_{identifier}.pkl"
         
         output_file = os.path.join(output_dir ,output_file)

         with open(output_file ,"wb") as f:
             pickle.dump(results_slim,f)

    return results_slim