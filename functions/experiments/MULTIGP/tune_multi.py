
import numpy as np
from sklearn.model_selection import KFold
from slim_gsgp_lib_np.main_multi_slim import multi_slim
from functions.metrics_test import *
from functions.experiments.GP.config_gp import *


def multi_tune(gen_params, 
               dataset, 
               split_id, 
               n_splits=5):
    
    params = gen_params.copy()
    dataset = dataset.copy()
    mask = dataset.pop('mask')
    params.pop('bcv_rmse', None)
    
    gp_params = params.pop('gp_params', {})
    for key in list(params.keys()):
        if 'gp' in key and key != 'gp_version':
            new_key = key.replace('_gp', '')
            gp_params[new_key] = params.pop(key)
    params['params_gp'] = gp_params

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses_tr, rmses_te, nodes, tnodes, divg = [], [], [], [], []
    best_ensemble_rmses, best_ensemble_sizes, norm_errs = [], [], []

    for i, (tr, te) in enumerate(kf.split(dataset['X_train'])):
        dataset_kf = dataset.copy()
        X_tr, y_tr = dataset['X_train'][tr], dataset['y_train'][tr]
        X_te, y_te = dataset['X_train'][te], dataset['y_train'][te]
        dataset_kf['X_train'] = X_tr
        dataset_kf['y_train'] = y_tr
        mask_kf = [sbmask[tr] for sbmask in mask]

        res = multi_slim(
            **params,
            **dataset_kf, 
            seed=split_id + i
        )
        elite, pop = res.elite, res.spec_pop

        min_errs, sizes = [], []
        ensemb_sqerr = 0
        for submask in mask_kf: 
            errors_mask = pop.errors_case[:, submask]
            errors_ind = np.sqrt(np.mean(errors_mask**2, axis=1))
            best_ind = np.argmin(errors_ind)
            min_err = errors_ind[best_ind]
            min_errs.append(min_err)
            sizes.append(pop.population[best_ind].total_nodes)
            ensemb_sqerr += np.sum(errors_mask[best_ind] ** 2)
        
        ensemb_sqerr = np.sqrt(ensemb_sqerr / len(mask_kf[0]))
        norm_err = np.sqrt(np.sum(np.array(min_errs)**2))
        rmse_train = rmse(elite.predict(X_tr), y_tr)
        rmse_test = rmse(elite.predict(X_te), y_te)

        best_ensemble_rmses.append(ensemb_sqerr)
        best_ensemble_sizes.append(np.sum(sizes))
        norm_errs.append(norm_err)
        rmses_tr.append(rmse_train)
        rmses_te.append(rmse_test)
        tnodes.append(elite.total_nodes)
        nodes.append(elite.nodes_count)
        divg.append(rmse_train/ensemb_sqerr)

    return float(np.mean(rmses_te)), {
        'std_rmse_elite'    : float(np.std(rmses_te)),              # Std of the elite ensemble tree
        'mean_tnodes_elite' : float(np.mean(tnodes)),               # Mean total nodes of the elite ensemble tree
        'std_nodes_elite'   : float(np.std(tnodes)),                # Std of the total nodes of the elite ensemble tree
        'mean_nodes'        : float(np.mean(nodes)),                # Mean total nodes of the ensemble
        'std_nodes'         : float(np.std(nodes)),                 # Std of the total nodes of the ensemble
        'norm_errs_ens'     : float(np.mean(norm_errs)),            # Mean of the normalized errors of the best ensemble achievable by the spec pop ensemble
        'sizes_spec_ens'    : float(np.mean(best_ensemble_sizes)),  # Mean of the sum of the sizes of the individuals that form the best ensemble
        'ensemble_rmse'     : float(np.mean(best_ensemble_rmses)),  # Rmse of the best ensemble achievable by the spec pop ensemble
        'rmse_train'        : float(np.mean(rmses_tr)),             # Mean rmse of the elite ensemble on the training set
        'divergence_tr'     : float(np.mean(divg)),                 # Divergence of the elite train rmse vs what could be the best one
    }





# -----------------------------------------------------------------------------------------------------------------------------------------------

def config_1(config): 
    config.update({ 
        'SPACE_PARAMETERS': config['SPACE_GP'],
        'N_SEARCHES_HYPER': config['N_SEARCHES_HYPER_GP'],
        'N_RANDOM_STARTS': config['N_RANDOM_STARTS_GP'],
        'selector' : config['SELECTOR_MULTI'],
    })
    return config 

def config_2(config, bp_1, task):
    config.update({ 
        'SPACE_PARAMETERS': config['SPACE_MULTI'],
        'N_SEARCHES_HYPER': config['N_SEARCHES_HYPER_MULTI'],
        'N_RANDOM_STARTS': config['N_RANDOM_STARTS_MULTI'],
        'selector' : config['SELECTOR_MULTI'],
    })
    config.update(bp_1)
    task['gen_params'].update(bp_1)

    return config, task