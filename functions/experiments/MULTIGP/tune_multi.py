
import numpy as np
import hashlib
from sklearn.model_selection import KFold
from slim_gsgp_lib_np.main_multi_slim import multi_slim
from functions.metrics_test import *
from functions.experiments.GP.config_gp import *

def multi_tune(gen_params, 
               dataset, 
               split_id,
               run, 
               n_splits=5):
    
    params = gen_params.copy()
    data_all = dataset.copy()
    mask = data_all.pop('mask')
    params.pop('bcv_rmse', None)
    
    gp_params = params.pop('gp_params', {})
    for key in list(params.keys()):
        if key.endswith('_gp') and key != 'gp_version':
            new_key = key.replace('_gp', '')
            gp_params[new_key] = params.pop(key)
    params['params_gp'] = gp_params

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses_tr, rmses_te, nodes, tnodes = [], [], [], []
    best_ensemble_rmses, best_ensemble_sizes, norm_errs = [], [], []

    for i, (idx_tr, idx_te) in enumerate(kf.split(data_all['X_train'])):
        X_tr, y_tr = data_all['X_train'][idx_tr], data_all['y_train'][idx_tr]
        X_te, y_te = data_all['X_train'][idx_te], data_all['y_train'][idx_te]
        mask_kf = [marr[idx_tr] for marr in mask] if mask is not None else None

        hash_Xtr = _hash_array(X_tr[:20])
        key = (split_id + i, hash_Xtr)

        if run == 'multi2' and key in _spec_pop_cache:
            # print(f"Using cached population for key: {key}")
            spec_pop = _spec_pop_cache[key]
        else:
            spec_pop = None

        res = multi_slim(
            **params,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            seed=split_id + i,
            population=spec_pop
        )
        elite, pop = res.elite, res.spec_pop

        if run == 'multi2':
            _spec_pop_cache[key] = pop

        if mask is not None: 
            min_errs, sizes = [], []
            ensemb_sqerr = 0
            for submask in mask_kf: 
                errors_mask = pop.errors_case[:, submask]
                errors_ind  = np.sqrt(np.mean(errors_mask**2, axis=1))
                best_ind    = np.argmin(errors_ind)
                min_errs.append(errors_ind[best_ind])
                sizes.append(pop.population[best_ind].total_nodes)
                ensemb_sqerr += np.sum(errors_mask[best_ind] ** 2)
            
            ensemb_sqerr = np.sqrt(ensemb_sqerr / len(mask_kf[0]))
            norm_err    = np.sqrt(np.sum(np.array(min_errs)**2))
            best_ensemble_rmses.append(ensemb_sqerr)
            best_ensemble_sizes.append(np.sum(sizes))
            norm_errs.append(norm_err)
        
        rmse_train  = rmse(elite.predict(X_tr), y_tr)
        rmse_test   = rmse(elite.predict(X_te), y_te)
        rmses_tr.append(rmse_train)
        rmses_te.append(rmse_test)
        tnodes.append(elite.total_nodes)
        nodes.append(elite.nodes_count)

    stats_general = {
        'mean_tnodes_elite' : float(np.mean(tnodes)),
        'mean_nodes'        : float(np.mean(nodes)),
        'rmse_train'        : float(np.mean(rmses_tr)),
    }

    if mask is not None: 
        stats_general.update({
            'sizes_spec_ens'    : float(np.mean(best_ensemble_sizes)),
            'norm_errs_ens'     : float(np.mean(norm_errs)),
            'ensemble_rmse'     : float(np.mean(best_ensemble_rmses)),
            'divergence_tr'     : float(np.mean(rmses_tr) / np.mean(best_ensemble_rmses)),
        })

    return float(np.mean(rmses_te)), stats_general, {
        'std_rmse_elite'    : float(np.std(rmses_te)),
        'std_nodes_elite'   : float(np.std(tnodes)),
        'std_nodes'         : float(np.std(nodes)),
    }



# ------------------------------------------------ CONFIG HANDLING -------------------------------------------------------------------------

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


# ------------------------------------------------ CACHING -------------------------------------------------------------------------
_spec_pop_cache = {}

def _hash_array(arr: np.ndarray) -> str:
    """Erzeuge einen MD5-Hash aus dem raw-Buffer eines NumPy-Arrays."""
    m = hashlib.md5()
    m.update(arr.tobytes())
    return m.hexdigest()

def _freeze(obj):
    """
    Wandelt ein beliebig verschachteltes Objekt in eine hashbare Form um:
    - dict  -> tuple(sorted((key, freeze(value)) ...))
    - list  -> tuple(freeze(elem) for elem in list)
    - set   -> tuple(sorted(freeze(elem) for elem in set))
    - andere Typen (int, float, str) bleiben unverändert
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, set):
        return tuple(sorted(_freeze(v) for v in obj))
    return obj

