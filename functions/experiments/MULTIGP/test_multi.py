import numpy as np
from slim_gsgp_lib_np.main_multi_slim import multi_slim
import time 

from functions.metrics_test import *
from functions.experiments.GP.config_gp import *
from functions.utils_test import simplify_tuple_expression
from functions.misc_functions import get_classification_summary


def multi_test(best_params, 
               dataset, 
               split_id, 
               seed):
    
    new_dict = {}
    for key in list(best_params.keys()):
        if '_gp' in key:
            new_key = key.replace('_gp', '')
            new_dict[new_key] = best_params.pop(key)
    best_params['params_gp'] = new_dict

    mask = dataset.pop('mask')
    bcv_rmse = best_params.pop('bcv_rmse')
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_test, y_test = dataset['X_test'], dataset['y_test']

    t0 = time.time()
    res = multi_slim(
        **dataset,
        **best_params, 
        seed=seed,
    )
    elapsed = time.time() - t0

    elite, pop, spec_pop, log = res.elite, res.population, res.spec_pop, res.log 
    pop_stats = [(rmse(ind.predict(X_test), y_test), ind.total_nodes, elapsed) for ind in pop]

    min_errs, sizes = [], [], []
    total_sq_errs = 0
    for submask in mask: 
        errors_mask = spec_pop.population.errors_case[:, submask]
        errors_ind = np.sqrt(np.mean(errors_mask**2, axis=1))
        best_ind = np.argmin(errors_ind)
        min_err = errors_ind[best_ind]
        min_errs.append(min_err)
        sizes.append(spec_pop.population[best_ind].total_nodes)
        total_sq_errs += np.sum(errors_mask[best_ind] ** 2)
    best_ensemble_possible = np.sqrt(total_sq_errs / mask.shape[1])

    classes = get_classification_summary(elite.collection, X_train, mask)

    y_test_pred      = elite.predict(X_test)
    y_train_pred     = elite.predict(X_train)   
    rmse_test        = rmse(y_test_pred, y_test)
    mae_test         = mae(y_test_pred, y_test)
    r2_test          = r_squared(y_test, y_test_pred)
    rmse_train       = rmse(y_train_pred, y_train)
    gen_gap          = 100 * abs(rmse_test - bcv_rmse) / bcv_rmse
    ensemble_gap     = 100 * (rmse_train - best_ensemble_possible) / rmse_train
    best_specialists = min_errs
    overfit          = 100 * (rmse_train - rmse_test) / rmse_train
    latex_repr       = simplify_tuple_expression(elite.repr_)

    records = { 
        'dataset_name'          : best_params['dataset_name'],
        'split_id'              : split_id,
        'trial_id'              : seed,
        'seed'                  : seed,
        'rmse_test'             : rmse_test,
        'mae_test'              : mae_test,
        'r2_test'               : r2_test,
        'gen_gap_per'           : gen_gap,
        'ensemble_gap_per'      : ensemble_gap,
        'best_specialists'      : best_specialists,
        'classes'               : classes,
        'nodes'                 : elite.total_nodes,
        'nodes_count'           : elite.nodes_count,
        'depth'                 : elite.depth,
        'train_rmse'            : rmse_train,
        'overfit_per'           : overfit,
        'time'                  : elapsed,
        'latex_repr'            : latex_repr,
    }

    return records, pop_stats, log