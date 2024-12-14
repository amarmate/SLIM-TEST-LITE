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
from slim_gsgp_lib.datasets.data_loader import *
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


if __name__ == '__main__':
    seed = 0
    datasets = [globals()[i] for i in globals() if 'load' in i][2:]
    X,y = datasets[10]()
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X = torch.tensor(scaler_X.fit_transform(X))
    y = torch.tensor(scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=seed)
    
    start = time.time()
    print('Single core performance test...')
    example_tree = slim(X_train=X_train, y_train=y_train, test_elite=False, dataset_name='test',
                        max_depth=22, init_depth=10, pop_size=200, n_iter=250, p_inflate=0.1, seed=seed,
                        struct_mutation=True, decay_rate=0.05, type_structure_mutation='new', timeout=200, verbose=0)
    end_single_core = time.time()
    print(f'Single core performance: {end_single_core-start}sec')

    print('Multi-core performance test...')
    start = time.time()
    with ProcessPoolExecutor() as executor:
        # Submete 5 instâncias paralelas
        futures = [executor.submit(slim, 
                                    X_train=X_train, y_train=y_train, test_elite=False, dataset_name='test',
                                    max_depth=22, init_depth=10, pop_size=200, n_iter=250, p_inflate=0.1, seed=seed,
                                    struct_mutation=True, decay_rate=0.05, type_structure_mutation='new', 
                                    timeout=200, verbose=0) 
                for _ in range(5)]
        
        # Apenas aguarda a conclusão de todas as tarefas
        for _ in as_completed(futures):
            pass  # Ignora os resultados

    end_multi_core = time.time()
    print(f'Multi-core performance: {end_multi_core - start} sec')
        
    