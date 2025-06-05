
import numpy as np
from sklearn.model_selection import KFold
from slim_gsgp_lib_np.main_gp import gp
from functions.metrics_test import *
from functions.experiments.GP.config_gp import *


def gp_tune(gen_params, 
            dataset,
            split_id, 
            n_splits=5,
            **kwargs):
    
    dataset.pop('mask')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses, nodes = [], []
    for i, (tr, te) in enumerate(kf.split(dataset['X_train'])):
        dataset_kf = dataset.copy()
        X_tr, y_tr = dataset['X_train'][tr], dataset['y_train'][tr]
        X_te, y_te = dataset['X_train'][te], dataset['y_train'][te]
        dataset_kf['X_train'] = X_tr
        dataset_kf['y_train'] = y_tr

        res = gp(**gen_params, 
                **dataset_kf,
                seed=split_id + i)
        
        elite = res.elite
        rmses.append(rmse(elite.predict(X_te), y_te))
        nodes.append(elite.total_nodes)

    return float(np.mean(rmses)), {
        'std_rmse': float(np.std(rmses)),
        'mean_nodes': float(np.mean(nodes)),
        'std_nodes': float(np.std(nodes)),
    }, {}