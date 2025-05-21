import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
from slim_gsgp_lib_np.datasets.data_loader import (
    load_airfoil, load_breast_cancer, load_concrete_strength, load_ld50
)
from functions.metrics_test import rmse
from functions.misc_functions import pf_rmse_comp_extended
import numpy as np
import pandas as pd

# Force single-threaded numeric libs
os.environ.update({k: '1' for k in [
    "OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"
]})

gen_params = { 
    "test_elite": True,
    "dataset_name": "test",
    "pop_size": 100,
    "n_iter": 2000,
    "max_depth": 9,
    "init_depth": 2,
    "p_xo": 0.8,
    "prob_const": 0.2,
    "prob_terminal": 0.7,
    "tree_functions": ['add','multiply','subtract','AQ'],
    "log_level": 'evaluate',
    "it_tolerance": 20000,
    "down_sampling": 1,
    "full_return": True,
    "verbose": False,
}

selectors = ['dalex', 'dalex_fast', 'dalex_fast_rand']

def create_split(loader, split_id):
    X, y = loader()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, p_test=0.2, seed=split_id
    )
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

def run_task(task):
    """
    task is a tuple: (dataset_name, data_split, split_id, seed, selector)
    """
    dataset_name, data_split, split_id, seed, selector = task
    t0 = time.time()

    optimizer = gp(
        seed=seed,
        **gen_params,
        **data_split,
        selector=selector,
        dalex_n_cases=5,
        tournament_size=2,
        particularity_pressure=10,
    )
    dt = time.time() - t0

    elite = optimizer.elite
    pop = optimizer.population
    hof = optimizer.hall_of_fame[:]  # Kopie der Liste
    log = optimizer.log

    # Hall-of-fame erweitern
    hof.extend(pop)
    points = [(ind.fitness, ind.total_nodes) for ind in hof]
    pf = pf_rmse_comp_extended(points)

    train_fit = elite.fitness
    test_fit = rmse(elite.predict(data_split['X_test']), data_split['y_test'])
    nodes_count = elite.total_nodes

    train_rmse = log['train_rmse']
    val_rmse = log['val_rmse']
    diversity_var = log['diversity_var']
    nodes_log = log['nodes_count']

    auc_train = np.trapezoid(train_rmse, dx=1)
    auc_val = np.trapezoid(val_rmse, dx=1)
    mean_diversity = np.mean(diversity_var)
    mean_size = np.mean(nodes_log)

    itconv_train = np.where(np.array(val_rmse) < test_fit)[0]
    itconv_test = np.where(np.array(train_rmse) < test_fit)[0]

    return {
        "dataset": dataset_name,
        "split_id": split_id,
        "seed": seed,
        "selector": selector,
        "train_rmse": train_fit,
        "test_rmse": test_fit,
        "nodes_count": nodes_count,
        "time_s": dt,
        "auc_train": auc_train,
        "auc_val": auc_val,
        "mean_diversity": mean_diversity,
        "mean_size": mean_size,
        "itconv_train": itconv_train,
        "itconv_test": itconv_test,
        "pf": pf
    }

if __name__ == "__main__":
    dataset_names = { 
        "airfoil": load_airfoil,
        "breast_cancer": load_breast_cancer,
        "concrete_strength": load_concrete_strength,
        "ld50": load_ld50
    }

    # Unveränderte Task-Liste
    tasks = [
        (name, create_split(loader, split_id), split_id, seed, selector)
        for name, loader in dataset_names.items()
        for split_id in range(3)
        for seed in range(1, 20)
        for selector in selectors
    ]

    n_cores = 16
    print(f"Starte auf {n_cores} Cores, insgesamt {len(tasks)} Tasks")

    results = []
    with Pool(processes=n_cores) as pool:
        for res in tqdm(
            pool.imap_unordered(run_task, tasks),
            total=len(tasks),
            desc="GP Experiments"
        ):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("gp_experiment_results.csv", index=False)
    print("Fertig – Ergebnisse in gp_experiment_results.csv")
