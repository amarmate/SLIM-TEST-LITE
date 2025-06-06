import numpy as np
from slim_gsgp_lib_np.main_multi_slim import multi_slim
import time 

from functions.metrics_test import *
from functions.experiments.GP.config_gp import *
from functions.utils_test import simplify_tuple_expression_multi
from functions.misc_functions import get_classification_summary

from slim_gsgp_lib_np.utils.callbacks import LogSpecialist


def multi_test(best_params, 
               dataset, 
               split_id, 
               seed):
    
    dataset = dataset.copy()
    params = best_params.copy()
    new_dict = {}
    for key in list(params.keys()):
        if '_gp' in key:
            new_key = key.replace('_gp', '')
            new_dict[new_key] = params.pop(key)

    params['params_gp'] = new_dict
    params['params_gp']['it_tolerance'] = 1

    mask = dataset.pop('mask', None) 
    bcv_rmse = params.pop('bcv_rmse')
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_test, y_test = dataset['X_test'], dataset['y_test']

    if mask is not None:
        l_spec = LogSpecialist(X_train, y_train, mask)
        params['params_gp']['callbacks'] = [l_spec]

    t0 = time.time()
    res = multi_slim(
        **dataset,
        **params, 
        seed=seed,
    )
    elapsed = time.time() - t0

    elite, pop, log = res.elite, res.population, res.log 
    spec_pop = res.spec_pop
    logs = [log, l_spec.get_log_dict()] if mask is not None else [log]
    pop_stats = [(rmse(ind.predict(X_test), y_test), ind.total_nodes, elapsed) for ind in pop]

    rmse_train       = rmse(y_train_pred, y_train)

    if mask is not None:
        min_errs, sizes = [], []
        total_sq_errs = 0
        for submask in mask: 
            errors_mask = spec_pop.errors_case[:, submask]
            errors_ind = np.sqrt(np.mean(errors_mask**2, axis=1))
            best_ind = np.argmin(errors_ind)
            min_err = errors_ind[best_ind]
            min_errs.append(min_err)
            sizes.append(spec_pop.population[best_ind].total_nodes)
            total_sq_errs += np.sum(errors_mask[best_ind] ** 2)
        best_ensemble_possible = np.sqrt(total_sq_errs / len(mask[0]))

        classes          = str(get_classification_summary(elite.collection, X_train, mask))
        best_specialists = min_errs
        ensemble_gap     = 100 * (rmse_train - best_ensemble_possible) / rmse_train

        records_mask = { 
            'best_specialists'      : best_specialists,
            'ensemble_gap_per'      : ensemble_gap,
            'classes'               : str(classes),
        }

    y_test_pred      = elite.predict(X_test)
    y_train_pred     = elite.predict(X_train)   
    rmse_test        = rmse(y_test_pred, y_test)
    mae_test         = mae(y_test_pred, y_test)
    r2_test          = r_squared(y_test, y_test_pred)
    gen_gap          = 100 * abs(rmse_test - bcv_rmse) / bcv_rmse
    overfit          = 100 * (rmse_train - rmse_test) / rmse_train
    latex_repr       = simplify_tuple_expression_multi(elite.collection)


    records = { 
        'dataset_name'          : best_params['dataset_name'],
        'split_id'              : split_id,
        'trial_id'              : seed,
        'seed'                  : seed,
        'rmse_test'             : rmse_test,
        'mae_test'              : mae_test,
        'r2_test'               : r2_test,
        'gen_gap_per'           : gen_gap,
        'nodes'                 : elite.total_nodes,
        'nodes_count'           : elite.nodes_count,
        'depth'                 : elite.depth,
        'train_rmse'            : rmse_train,
        'overfit_per'           : overfit,
        'time'                  : elapsed,
        'latex_repr'            : latex_repr,
    }

    if mask is not None:
        records.update(records_mask)

    return records, pop_stats, logs 