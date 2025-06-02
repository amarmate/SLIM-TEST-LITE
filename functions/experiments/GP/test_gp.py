import numpy as np
from slim_gsgp_lib_np.main_gp import gp
import time 

from functions.metrics_test import *
from functions.experiments.GP.config_gp import *
from functions.utils_test import simplify_tuple_expression


def gp_test(best_params, split_id, seed):
    params = best_params.copy()
    bcv_rmse = params.pop('bcv_rmse')
    X_train, y_train = params['X_train'], params['y_train']
    X_test, y_test = params['X_test'], params['y_test']

    t0 = time.time()
    res = gp(
        **params, 
        seed=seed,
    )
    elapsed = time.time() - t0

    elite, pop, log = res.elite, res.population, res.log 
    pop_stats = [(rmse(ind.predict(X_test), y_test), ind.total_nodes, elapsed) for ind in pop]

    y_test_pred  = elite.predict(X_test)
    y_train_pred = elite.predict(X_train)   
    rmse_test    = rmse(y_test_pred, y_test)
    mae_test     = mae(y_test_pred, y_test)
    r2_test      = r_squared(y_test, y_test_pred)
    rmse_train   = rmse(y_train_pred, y_train)
    gen_gap      = 100 * abs(rmse_test - bcv_rmse) / bcv_rmse
    overfit      = 100 * (rmse_train - rmse_test) / rmse_train
    latex_repr   = simplify_tuple_expression(elite.repr_)

    records = { 
        'dataset_name'          : params['dataset_name'],
        'split_id'              : split_id,
        'trial_id'              : seed,  # Using seed as trial_id for simplicity
        'seed'                  : seed,
        'rmse_test'             : rmse_test,
        'mae_test'              : mae_test,
        'r2_test'               : r2_test,
        'gen_gap_%'             : gen_gap,
        'nodes'                 : elite.total_nodes,
        'depth'                 : elite.depth,
        'train_rmse'            : rmse_train,
        'overfit_%'             : overfit,
        'time'                  : elapsed,
        'latex_repr'            : latex_repr,
    }


    return records, pop_stats, log