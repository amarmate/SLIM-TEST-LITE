import numpy as np
import matplotlib
matplotlib.use('Agg')

import time
import pandas as pd
import mlflow
from pathlib import Path

import pickle
from pathlib import Path

from slim_gsgp_lib_np.main_gp import gp
from functions.utils_test import pf_rmse_comp
from functions.metrics_test import *
from sklearn.model_selection import KFold
from skopt import gp_minimize
from functions.experiments.GP.config_gp import *



def tuning(config, 
        data_split, name, split_id, selector
        ): 
    
    trial_results, calls_count = [], 0

    temp_dir = Path('temp') / f"{EXPERIMENT_NAME}_{selector}" / name / f"split_{split_id}" / "tuning"
    temp_dir.mkdir(parents=True, exist_ok=True)
    trial_results, calls_count = [], 0

    X, _, y, _ = data_split

    def objective(params): 
        nonlocal calls_count
        # md, px, pc, pt, pp, dnc = params
        md, pis, pp, pxo = params
        ngen, pop_size = PI[pis]
        t0 = time.time()

        kf = KFold(n_splits=N_CV, shuffle=True, random_state=SEED)
        rmses, nodes, times, all_fold_pf = [], [], [], []
        
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            talg = time.time()
            res = gp(X_train=X_train, y_train=y_train, test_elite=False, dataset_name=name,
                        pop_size=int(pop_size), n_iter=int(ngen), selector=selector,
                        max_depth=int(md), init_depth=2, p_xo=pxo, prob_const=0.2, 
                        prob_terminal=0.7, particularity_pressure=pp, seed=SEED+i,
                        full_return=True, n_jobs=1, verbose=False, log_level=0,
                        tree_functions=FUNCTIONS, it_tolerance=STOP_THRESHOLD, 
                        down_sampling=1,
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
            'init_depth':              2,
            'max_depth':               int(md),
            'p_xo':                    pxo,
            'prob_const':              0.2,
            'pop_size':                pop_size,
            'n_iter':                  ngen,
            'prob_terminal':           0.7,
            'particularity_pressure':  pp,
            # 'dalex_n_cases':          int(dnc),
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

    df_tr.to_pickle(temp_dir / "checkpoint_tuning.pkl")
    with open(temp_dir / "checkpoint_best_params.pkl", 'wb') as f:
        pickle.dump(best_hyperparams, f)

    return df_tr, best_hyperparams, best_cv_rmse 

