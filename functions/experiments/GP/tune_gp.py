
import numpy as np
from sklearn.model_selection import KFold
from slim_gsgp_lib_np.main_gp import gp
from functions.metrics_test import *
from functions.experiments.GP.config_gp import *


def gp_tune(gp_params, split_id, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses, nodes = [], []
    for i, (tr, te) in enumerate(kf.split(gp_params['X_train'])):
        copy_params = gp_params.copy()
        X_tr, y_tr = gp_params['X_train'][tr], gp_params['y_train'][tr]
        X_te, y_te = gp_params['X_train'][te], gp_params['y_train'][te]
        copy_params['X_train'] = X_tr
        copy_params['y_train'] = y_tr
        copy_params['max_depth'] = int(copy_params['max_depth'])

        res = gp(**copy_params, seed=split_id + i)
        elite = res.elite

        rmses.append(rmse(elite.predict(X_te), y_te))
        nodes.append(elite.total_nodes)

    return float(np.mean(rmses)), {
        'std_rmse': float(np.std(rmses)),
        'mean_nodes': float(np.mean(nodes)),
        'std_nodes': float(np.std(nodes)),
    }