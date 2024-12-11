from slim_gsgp_lib.main_slim import slim
from slim_gsgp_lib.utils.utils import train_test_split
from slim_gsgp_lib.evaluators.fitness_functions import rmse
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import time

from slim_gsgp_lib.algorithms.SLIM_GSGP.operators.mutators import *
from slim_gsgp_lib.utils.utils import *
from functions.test_algorithms import *
from functions.random_search import * 
from slim_gsgp_lib.datasets.data_loader import *


# DISCOVER WHY THIS IS TAKING SO LONG!

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

X,y = datasets[10]()
# Scale
scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
X = torch.tensor(scaler_X.fit_transform(X))
y = torch.tensor(scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1))
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2)

start = time.time()

example_tree = slim(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    max_depth=22, init_depth=10, pop_size=100, n_iter=250, p_inflate=0.1, seed=0,
                    struct_mutation=True, decay_rate=0.05, type_structure_mutation='new', timeout=100)

print(f'{time.time()-start}sec)')
