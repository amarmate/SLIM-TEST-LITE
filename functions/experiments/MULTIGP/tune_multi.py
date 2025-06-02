
import numpy as np
from sklearn.model_selection import KFold
from slim_gsgp_lib_np.main_multi_slim import multi_slim
from functions.metrics_test import *
from functions.experiments.GP.config_gp import *


def multi_tune(params, split_id, n_splits=5):
    mask = params.pop('mask')
    new_dict = {}
    for key in list(params.keys()):
        if 'gp' in key:
            new_key = key.replace('_gp', '')
            new_dict[new_key] = params.pop(key)
    params['params_gp'] = new_dict

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses, nodes = [], []
    for i, (tr, te) in enumerate(kf.split(params['X_train'])):
        copy_params = params.copy()
        X_tr, y_tr = params['X_train'][tr], params['y_train'][tr]
        X_te, y_te = params['X_train'][te], params['y_train'][te]
        copy_params['X_train'] = X_tr
        copy_params['y_train'] = y_tr

        res = multi_slim(**copy_params, seed=split_id + i)
        elite, pop = res.elite, res.population

        min_errs, sizes = [], [], []
        total_sq_errs = 0
        for submask in mask: 
            errors_mask = pop.population.errors_case[:, submask]
            errors_ind = np.sqrt(np.mean(errors_mask**2, axis=1))
            best_ind = np.argmin(errors_ind)
            min_err = errors_ind[best_ind]
            min_errs.append(min_err)
            sizes.append(pop.population[best_ind].total_nodes)
            total_sq_errs += np.sum(errors_mask[best_ind] ** 2)
        
        total_sq_errs = np.sqrt(total_sq_errs / mask.shape[1])

        rmses.append(rmse(elite.predict(X_te), y_te))
        nodes.append(elite.total_nodes)

    return float(np.mean(rmses)), {
        'std_rmse': float(np.std(rmses)),
        'mean_nodes': float(np.mean(nodes)),
        'std_nodes': float(np.std(nodes)),
        'min_errs_spec' : min_errs,
        'sizes_spec' : sizes,
        'ensemble_rmse' : total_sq_errs,
        'ensemble_size' : np.sum(sizes),
    }


def config_1(config): 
    config.update({ 
        'SPACE_PARAMETERS': config['SPACE_GP'],
        'N_SEARCHES_HYPER': config['N_SEARCHES_HYPER_GP'],
        'N_RANDOM_STARTS': config['N_RANDOM_STARTS_GP'],
        'selector' : config['SELECTOR_GP'],
        'PI' : config['PI_GP'],
    })
    return config 

def config_2(config, bp_1):
    config.update({ 
        'SPACE_PARAMETERS': config['SPACE_MULTI'],
        'N_SEARCHES_HYPER': config['N_SEARCHES_HYPER_MULTI'],
        'N_RANDOM_STARTS': config['N_RANDOM_STARTS_MULTI'],
        'selector' : config['SELECTOR_MULTI'],
        'PI' : config['PI_MULTI'],
    })
    config.update(bp_1)

    return config