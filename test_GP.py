import os
import argparse
import pickle
import time 
import pandas as pd
from joblib import Parallel, delayed

from slim_gsgp_lib_np.main_gp import gp

from functions.utils import pf_rmse_comp, pf_rmse_comp_time, save_tuning_results
from functions.test_algorithms import *
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Integer, Real

# ------------------------------------------------   SETTINGS   --------------------------------------------------------------
N_SPLITS = 3
N_CV = 4
N_SEARCHES_HYPER = 20
N_RANDOM_STARTS = 10
NOISE_SKOPT = 1e-3
N_TESTS = 8
P_TEST = 0.2 
SEED = 20
POP_SIZE = 100
N_GENERATIONS = 2000
SELECTOR = 'dalex'
np.random.seed(SEED)

SPACE_PARAMETERS = [
        Integer(2, 4, name='init_depth'),
        Integer(4, 8, name='max_depth'),                   
        Real(0.6, 0.9, name='p_xo'),                                          
        Real(0.1, 0.25, name='prob_const'),                                       
        Real(0.6, 0.9, name='prob_terminal'),                                      
        Real(10, 50, name='particularity_pressure'),
]

# ------------------------------------------- LIMIT THREADS FOR NUMPY --------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

# ----------------------------------------------------- DATASETS --------------------------------------------------------------
from slim_gsgp_lib_np.datasets.synthetic_datasets import (
    load_synthetic1, load_synthetic2, load_synthetic3, load_synthetic4, load_synthetic5,
    load_synthetic6, load_synthetic7, load_synthetic8, load_synthetic9, load_synthetic10, load_synthetic11,
)
from slim_gsgp_lib_np.datasets.data_loader import ( 
    load_airfoil, load_boston, load_concrete_strength, load_diabetes, load_efficiency_heating, load_forest_fires,
    load_istanbul, load_ld50, load_bioav, load_parkinson_updrs, load_ppb, load_resid_build_sale_price,
)

datasets = {
    name.split('load_')[1] : loader for name, loader in globals().items() if name.startswith('load_') and callable(loader)
}

# ----------------------------------------------------- MAIN FUNCTIONS -------------------------------------------------------------
def tuning(data_split, name, split_id): 
    trial_results, calls_count = [], 0

    def objective(params): 
        nonlocal calls_count
        id, md, px, pc, pt, pp = params
        t0 = time.time()

        kf = KFold(n_splits=N_CV, shuffle=True, random_state=SEED)
        rmses, nodes, times, all_fold_pf = [], [], [], []
        
        for i, (train_index, test_index) in enumerate(kf.split(data_split[0])):
            X_train, X_val = data_split[0][train_index], data_split[0][test_index]
            y_train, y_val = data_split[1][train_index], data_split[1][test_index]

            talg = time.time()
            res = gp(X_train=X_train, y_train=y_train, test_elite=False, dataset_name=name,
                        pop_size=POP_SIZE, n_iter=N_GENERATIONS, selector=SELECTOR,
                        max_depth=md, init_depth=id, p_xo=px, prob_const=pc, 
                        prob_terminal=pt, particularity_pressure=pp, seed=SEED+i,
                        full_return=True, n_jobs=1, verbose=False, log_level=0,
                        tree_functions=['add', 'multiply', 'divide', 'sqrt'], 
            )
            elite, population = res
            rmses.append(rmse(elite.predict(X_val), y_val))
            nodes.append(elite.total_nodes)
            times.append(time.time() - talg)
            pop_stats = [(rmse(ind.predict(X_val), y_val), ind.total_nodes) for ind in population]
            all_fold_pf.extend(pop_stats)
        
        mean_rmse    = float(np.mean(rmses))
        std_rmse     = float(np.std(rmses))
        mean_nodes   = float(np.mean(nodes))
        std_time     = float(np.std(times))
        elapsed      = time.time() - t0
        pop_stats_pf = pf_rmse_comp(pop_stats)
        calls_count += 1

        trial_results.append({
            'dataset_name':            name,
            'split_id':                split_id,
            'trial_id':                calls_count,
            'init_depth':              id,
            'max_depth':               md,
            'p_xo':                    px,
            'prob_const':              pc,
            'prob_terminal':           pt,
            'particularity_pressure':  pp,
            'mean_cv_rmse':            mean_rmse,
            'std_cv_rmse':             std_rmse,
            'mean_cv_nodes':           mean_nodes,
            'cv_elapsed_sec':          elapsed,
            'std_time':                std_time,  
            'pareto_front':            pop_stats_pf,
        })

        return np.mean(rmses)

    # OPTIMIZATION
    gp_minimize(
        func=lambda params: objective(params),
        dimensions=SPACE_PARAMETERS,
        n_calls=N_SEARCHES_HYPER,
        noise=NOISE_SKOPT,
        n_initial_points=N_RANDOM_STARTS,
        verbose=False,
        random_state=SEED,
    )

    df_tr = pd.DataFrame(trial_results)
    best_trial = df_tr.loc[df_tr['mean_cv_rmse'].idxmin()]
    best_hyperparams = best_trial[['init_depth','max_depth','p_xo','prob_const',
                                   'prob_terminal','particularity_pressure']].to_dict()

    save_tuning_results(name, split_id, df_tr, best_hyperparams)
    return best_hyperparams


def test_algo(): 
    # TESTING 

    # SAVING 
    pass

def parse_args(): 
    parser = argparse.ArgumentParser(description='Run GP experiments')
    parser.add_argument('--max_workers', type=int, default=0, help='Number of parallel workers (default: all available)')
    if parser.parse_args().max_workers > os.cpu_count(): 
        raise ValueError(f"max_workers cannot be greater than the number of available CPU cores ({os.cpu_count()})")
    
    return parser.parse_args()

def run_experiment(dataset, name): 
    X, y = dataset()
    for i in range(N_SPLITS): 
        data_split = train_test_split(X, y, p_test=P_TEST, seed=SEED+i)
        params = tuning(data_split, name, i)
        test_algo(params, data_split, name, i)

if __name__ == '__main__': 
    args = parse_args()
    parallel_jobs = args.max_workers if args.max_workers > 0 else os.cpu_count()
    print(f"Running trials with {parallel_jobs} parallel jobs...")

    # Execute trials in parallel
    Parallel(n_jobs=parallel_jobs)(
        delayed(run_experiment)(dataset, name) for name, dataset in datasets.items()
        )
