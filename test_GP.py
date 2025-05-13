import os
import argparse
import time 
import pandas as pd
from joblib import Parallel, delayed
import mlflow

from slim_gsgp_lib_np.main_gp import gp
from functions.utils import (pf_rmse_comp, pf_rmse_comp_time, 
                             save_tuning_results, save_test_results, save_experiment_results,
                             simplify_tuple_expression, register_mlflow_charts, log_latex_as_image)

from functions.test_algorithms import *
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Integer, Real

# ------------------------------------------------   SETTINGS   --------------------------------------------------------------
N_SPLITS = 4
N_CV = 4
N_SEARCHES_HYPER = 25
N_RANDOM_STARTS = 10
NOISE_SKOPT = 1e-3
N_TESTS = 15
P_TEST = 0.2 
SEED = 20
POP_SIZE = 100
N_GENERATIONS = 2500
SELECTOR = 'dalex'
N_TIME_BINS = 300
SUFFIX_SAVE = '1'
PREFIX_SAVE = 'GP'
EXPERIMENT_NAME = 'GP_Experiment'
FUNCTIONS = ['add', 'multiply', 'divide', 'subtract']
STOP_THRESHOLD = 400

np.random.seed(SEED)

SPACE_PARAMETERS = [
        Integer(2, 4, name='init_depth'),
        Integer(4, 9, name='max_depth'),                   
        Real(0.6, 0.9, name='p_xo'),                                          
        Real(0.1, 0.25, name='prob_const'),                                       
        Real(0.6, 0.9, name='prob_terminal'),                                      
        Real(10, 100, name='particularity_pressure', prior='log-uniform'),
]

# ------------------------------------------- LIMIT THREADS FOR NUMPY --------------------------------------------------------
os.environ.update({
    k: '1' for k in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
    ]})

# ----------------------------------------------------- DATASETS --------------------------------------------------------------
from slim_gsgp_lib_np.datasets.synthetic_datasets import (
    load_synthetic1, load_synthetic2, load_synthetic3, load_synthetic4, load_synthetic5,
    load_synthetic6, load_synthetic7, load_synthetic8, load_synthetic9, load_synthetic10, load_synthetic11,
)
from slim_gsgp_lib_np.datasets.data_loader import ( 
    load_airfoil, load_boston, load_concrete_strength, load_diabetes, load_efficiency_heating, load_forest_fires,
    load_istanbul, load_ld50, load_bioav, load_parkinson_updrs, load_ppb, load_resid_build_sale_price,
)

datasets = {name.split('load_')[1] : loader for name, loader in globals().items() if name.startswith('load_') and callable(loader)}

# ----------------------------------------------------- MAIN FUNCTIONS -------------------------------------------------------------
def tuning(data_split, name, split_id): 
    trial_results, calls_count = [], 0
    X, _, y, _ = data_split

    def objective(params): 
        nonlocal calls_count
        id, md, px, pc, pt, pp = params
        t0 = time.time()

        kf = KFold(n_splits=N_CV, shuffle=True, random_state=SEED)
        rmses, nodes, times, all_fold_pf = [], [], [], []
        
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            talg = time.time()
            res = gp(X_train=X_train, y_train=y_train, test_elite=False, dataset_name=name,
                        pop_size=POP_SIZE, n_iter=N_GENERATIONS, selector=SELECTOR,
                        max_depth=int(md), init_depth=int(id), p_xo=px, prob_const=pc, 
                        prob_terminal=pt, particularity_pressure=pp, seed=SEED+i,
                        full_return=True, n_jobs=1, verbose=False, log_level=0,
                        tree_functions=FUNCTIONS, it_tolerance=STOP_THRESHOLD, 
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

        mlflow.set_tag("tuning_step", calls_count)
        mlflow.log_metric("tuning_time", elapsed, step=calls_count)   
        mlflow.log_metric("tuning_rmse", mean_rmse, step=calls_count)
        mlflow.log_metric("tuning_nodes", mean_nodes, step=calls_count)

        trial_results.append({
            'dataset_name':            name,
            'split_id':                split_id,
            'trial_id':                calls_count,
            'seed':                    SEED,
            'init_depth':              int(id),
            'max_depth':               int(md),
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
    best_cv_rmse = best_trial['mean_cv_rmse']
    mlflow.log_params(best_hyperparams)
    mlflow.set_tag("tuning_step", 'complete')

    save_tuning_results(name, split_id, df_tr, best_hyperparams, prefix=PREFIX_SAVE, suffix=SUFFIX_SAVE)
    return best_hyperparams, best_cv_rmse


def test_algo(params, data_split, name, split_id, bcv_rmse): 
    X_train, X_test, y_train, y_test = data_split
    all_fold_pf, logs, records = [], [], []
    
    # TESTING 
    for test_n in range(N_TESTS): 
        talg = time.time()
        res = gp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_elite=True, dataset_name=name,
                    pop_size=POP_SIZE, n_iter=N_GENERATIONS, selector=SELECTOR, seed=SEED+test_n,
                    full_return=True, n_jobs=1, verbose=False, log_level='evaluate', **params,
                    tree_functions=FUNCTIONS, it_tolerance=np.inf, 
        )
        elapsed = time.time() - talg
        elite, population, log = res
        pop_stats = [(rmse(ind.predict(X_test), y_test), ind.total_nodes, elapsed) for ind in population]
        all_fold_pf.extend(pop_stats)
        logs.append(log)

        y_test_pred  = elite.predict(X_test)
        y_train_pred = elite.predict(X_train)   
        rmse_test    = rmse(y_test_pred, y_test)
        mae_test     = mae(y_test_pred, y_test)
        r2_test      = r_squared(y_test, y_test_pred)
        rmse_train   = rmse(y_train_pred, y_train)
        gen_gap      = 100 * abs(rmse_test - bcv_rmse) / bcv_rmse
        overfit      = 100 * (rmse_train - rmse_test) / rmse_train
        latex_repr   = simplify_tuple_expression(elite.repr_)

        records.append({
            'dataset_name':          name,
            'split_id':              split_id,
            'trial_id':              test_n,
            'seed':                  SEED,
            'rmse_test':             rmse_test,
            'mae_test':              mae_test,
            'r2_test':               r2_test,
            'generalization_gap_%':  gen_gap,
            'total_nodes':           elite.total_nodes,
            'depth':                 elite.depth,
            'train_rmse':            rmse_train,
            'overfitting_%':         overfit,
            'time':                  elapsed,
            'latex_repr':            latex_repr,
        })

        mlflow.set_tag("testing_step", test_n+1)
        mlflow.log_metric("testing_gen_gap", gen_gap, step=test_n+1)
        mlflow.log_metric("testing_overfitting", overfit, step=test_n+1)
        mlflow.log_metric("testing_rmse", rmse_test, step=test_n+1)    
        mlflow.log_metric("testing_nodes", elite.total_nodes, step=test_n+1)
        mlflow.log_metric("testing_time", elapsed, step=test_n+1)

    # SAVING 
    main_test = pd.DataFrame(records)
    save_test_results(name, split_id, main_test, prefix=PREFIX_SAVE, suffix=SUFFIX_SAVE)

    best_example = main_test.loc[main_test['rmse_test'].idxmin()]['latex_repr']
    log_latex_as_image(best_example, name, split_id, prefix=PREFIX_SAVE, suffix=SUFFIX_SAVE)

    pf = pf_rmse_comp_time(all_fold_pf)
    mlflow.set_tag("testing_step", 'complete')
    return pf, logs

def parse_args(): 
    parser = argparse.ArgumentParser(description='Run GP experiments')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (default: all available)')
    args = parser.parse_args()
    if args.workers > os.cpu_count():
        raise ValueError(f"workers cannot be greater than available CPU cores ({os.cpu_count()})")
    return args

def run_experiment(dataset, name): 
    print(f"Starting experiment for {name}...")
    pfs, logs = [], []
    mlflow.set_experiment(name)
    for i in range(N_SPLITS): 
        with mlflow.start_run(run_name=f'Split {i+1}', nested=True):
            mlflow.set_tag("dataset_name", name)
            data_split = train_test_split(dataset()[0], dataset()[1], p_test=P_TEST, seed=SEED + i)
            params, bcv_rmse = tuning(data_split, name, i)
            pf, log = test_algo(params, data_split, name, i, bcv_rmse)

        pfs.extend(pf)
        logs.extend(log)

    log_results = save_experiment_results(name, pfs, logs, n_bins=N_TIME_BINS, 
                            prefix=PREFIX_SAVE, suffix=SUFFIX_SAVE)
    
    with mlflow.start_run(run_name='Final Results', nested=True):
        register_mlflow_charts(name, log_results, pfs=pfs, prefix=PREFIX_SAVE, suffix=SUFFIX_SAVE)
        

if __name__ == '__main__': 
    args = parse_args()
    parallel_jobs = args.workers if args.workers > 0 else os.cpu_count()
    print(f"Running trials with {parallel_jobs} parallel jobs...")

    # Execute trials in parallel
    Parallel(n_jobs=parallel_jobs)(
        delayed(run_experiment)(dataset, name) for name, dataset in datasets.items()
        )
