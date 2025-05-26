import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
from slim_gsgp_lib_np.datasets.data_loader import (
    load_airfoil, load_breast_cancer, load_concrete_strength, load_ld50, load_bioav, load_boston, 
    load_efficiency_heating, load_istanbul, load_resid_build_sale_price, load_ppb
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

CUTOFF = 200
selectors = ['dalex', 'dalex_fast', 'dalex_fast_rand']

# ----------------------------------------------------------------------------------------------------------------------------------------------------
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

    best_test = min(val_rmse)
    best_train = min(train_rmse)
    itconv_train = np.where(np.array(val_rmse) == best_test)[0][0]
    itconv_test = np.where(np.array(train_rmse) == best_train)[0][0]

    last_best, count = np.inf, 0
    for i, point in enumerate(train_rmse): 
        if point < last_best * 0.999:
            count = 0
            last_best = point
        else:
            count += 1
        if count >= CUTOFF:
            break

    cutoff = i 
    cutoff_train = train_rmse[cutoff]
    cutoff_test = val_rmse[cutoff]

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
        "best_test": best_test,
        "cutoff_it": cutoff,
        "cutoff_train": cutoff_train,
        "cutoff_test": cutoff_test,
        "pf": pf
    }

def commit_and_push_data(filename, commit_msg):
    os.system(f"git add {filename}")
    os.system(f"git commit -m \"{commit_msg}\"")
    os.system("git push origin main")

if __name__ == "__main__":
    dataset_names = {
        "airfoil": load_airfoil,
        "breast_cancer": load_breast_cancer,
        "concrete_strength": load_concrete_strength,
        "ld50": load_ld50,
        "bioav": load_bioav,
        "boston": load_boston,
        "efficiency_heating": load_efficiency_heating,
        "istanbul": load_istanbul,
        "resid_build_sale_price": load_resid_build_sale_price,
        "ppb": load_ppb
    }

    partial_csv = os.path.join("/data", "gp_partial.csv")
    final_csv   = os.path.join("/data", "gp_experiment_results.csv")

    # 1) Data-Repo klonen oder updaten
    os.chdir(os.path.join('..', "/data"))
    if not os.path.isdir(".git"):
        os.system(f"git clone git@github.com:amarmate/data_transfer.git")
    else:
        os.system("git fetch origin && git reset --hard origin/main")

    # 2) Tasks aufbauen
    tasks = [
        (name, create_split(loader, split_id), split_id, seed, selector)
        for name, loader in dataset_names.items()
        for split_id in range(3)
        for seed in range(1, 20)
        for selector in selectors
    ]

    results = []
    # 3) Experimente in /SLIM ausführen
    os.chdir(os.path.join('..', '/SLIM'))
    with Pool(processes=min(16, os.cpu_count())) as pool:
        for i, res in enumerate(tqdm(pool.imap_unordered(run_task, tasks),
                                     total=len(tasks),
                                     desc="GP Experiments")):
            results.append(res)

            # Teilergebnisse alle 50 Tasks oder am Ende
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                df_part = pd.DataFrame(results)
                df_part.to_csv(partial_csv, index=False)

                os.chdir(os.path.join('..', "/data"))
                msg = f"Partial after {i+1} tasks @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
                commit_and_push_data("gp_partial.csv", msg)
                os.chdir(os.path.join('..', '/SLIM'))

    # 4) Finale Ergebnisse speichern und pushen
    df_full = pd.DataFrame(results)
    df_full.to_csv(final_csv, index=False)

    os.chdir(os.path.join('..', "/data"))
    commit_and_push_data("gp_experiment_results.csv",
                         f"Final results @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    print("Fertig – alle Ergebnisse in /data gespeichert und gepusht.")
