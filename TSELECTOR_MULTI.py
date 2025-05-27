import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
from slim_gsgp_lib_np.utils.callbacks import LogSpecialist
from slim_gsgp_lib_np.datasets.synthetic_datasets import (
    load_synthetic2, load_synthetic3, load_synthetic4,
    load_synthetic5, load_synthetic6, load_synthetic7,
    load_synthetic8, load_synthetic9, load_synthetic10, 
    load_synthetic11, load_synthetic12,
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

PARTIAL_NAME = "sel_multi_partial.csv"
FINAL_NAME = "sel_multi_results.csv"

gen_params = { 
    "test_elite": True,
    "dataset_name": "test",
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

PI_SETTINGS = [(1000, 250), (400, 500), (150, 1000)]  # (ITERATIONS, POP_SIZE)    
CUTOFF = 0.1

selectors = ['dalex', 'dalex_fast', 'dalex_fast_rand']

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def create_split(loader, split_id):
    X, y, _, mask = loader()
    train_ids, test_ids = train_test_split(
        X, y, p_test=0.2, seed=split_id, indices_only=True
    )
    X_train, X_test = X[train_ids], X[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
    mask_train = [sbmask[train_ids] for sbmask in mask]
    mask_test = [sbmask[test_ids] for sbmask in mask]
    dataset =  {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }
    masks = { 
        'mask_train': np.array(mask_train),
        'mask_test': np.array(mask_test),
    }
    return (dataset, masks)

def run_task(task):
    """
    task is a tuple: (dataset_name, data_split, mask_split, split_id, seed, selector, pi_setting)
    """
    dataset_name, data_split, mask_split, split_id, seed, selector, pi_setting = task
    t0 = time.time()
    lspec = LogSpecialist(
        data_split['X_train'], data_split['y_train'], mask_split['mask_train']
    )

    data_split.update({
        'n_iter': PI_SETTINGS[pi_setting][0],
        'pop_size': PI_SETTINGS[pi_setting][1],
    })

    optimizer = gp(
        seed=seed,
        **gen_params,
        **data_split,
        selector=selector,
        dalex_n_cases=5,
        tournament_size=2,
        particularity_pressure=10,
        callbacks=[lspec],
    )
    dt = time.time() - t0

    elite = optimizer.elite
    pop = optimizer.population
    hof = optimizer.hall_of_fame[:] 
    log = optimizer.log

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
    itconv_train = np.where(np.array(train_rmse) == best_train)[0][0]
    itconv_test = np.where(np.array(val_rmse) == best_test)[0][0]


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

    general_info = {
        "dataset": dataset_name,
        "split_id": split_id,
        "seed": seed,
        "selector": selector,
        "pop_size": data_split['pop_size'],
        "n_iter": data_split['n_iter'],
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

    l_rmse, l_size, l_rmse_out, l_ensemble = lspec.log_rmse, lspec.log_size, lspec.log_rmse_out, lspec.log_best_ensemble
    final_rmses = l_rmse[-1]
    final_sizes = l_size[-1]
    final_rmses_out = l_rmse_out[-1]

    final_ensemble = l_ensemble[-1]
    size_ensemble = np.sum(final_sizes)
    best_ensemble = min(l_ensemble)
    auc_ensemble = np.trapezoid(l_ensemble, dx=1)
    conv_ensemble = np.where(np.array(l_ensemble) == best_ensemble)[0][0]

    specific_spec_info = { 
        "rmses_spec": final_rmses,
        "sizes_spec": final_sizes,
        "rmseout_spec": final_rmses_out,
        "rmse_ensemble": final_ensemble,
        "size_ensemble": size_ensemble,
        "best_ensemble": best_ensemble,
        "auc_ensemble": auc_ensemble,
        "conv_ensemble": conv_ensemble,
    }

    general_info.update(specific_spec_info)
    return general_info

def commit_and_push_data(filename, commit_msg):
    os.system(f"git add {filename}")
    os.system(f"git commit -m \"{commit_msg}\"")
    os.system("git push origin main")


if __name__ == "__main__":
    dataset_names = {
        loader.__name__.split('_')[1]: loader
        for loader in [
            load_synthetic2, load_synthetic3, load_synthetic4,
            load_synthetic5, load_synthetic6, load_synthetic7,
            load_synthetic8, load_synthetic9, load_synthetic10,
            load_synthetic11, load_synthetic12
        ]
    }

    partial_csv = os.path.join("/data", PARTIAL_NAME)
    final_csv   = os.path.join("/data", FINAL_NAME)

    # 0) Bereits existierende Teilergebnisse einlesen
    results = []
    start_idx = 0
    if os.path.exists(partial_csv):
        df_exist = pd.read_csv(partial_csv)
        results = df_exist.to_dict('records')
        start_idx = len(df_exist)
        print(f"Überspringe {start_idx} bereits berechnete Tasks.")


    # 1) Data-Repo klonen oder updaten
    if not os.path.isdir(".git"):
        os.system(f"git clone git@github.com:amarmate/data_transfer.git")
    else:
        os.system("git fetch origin && git reset --hard origin/main")

    # Identify myself github 
    os.system("git config user.name 'Mateus GP Bot'")
    os.system("git config user.email 'mbaptistaamaral@gmail.com'")

    # 2) Tasks aufbauen
    tasks = [
        (name, *create_split(loader, split_id), split_id, seed, selector, pi_setting)
        for name, loader in dataset_names.items()
        for split_id in range(3)
        for seed in range(1, 20)
        for selector in selectors
        for pi_setting in range(0,3)
    ]

    # 3) Experimente in /SLIM ausführen
    os.chdir(os.path.join('..', '/SLIM'))
    tasks_to_run = tasks[start_idx:]

    with Pool(processes=min(16, os.cpu_count())) as pool:
        for rel_i, res in enumerate(tqdm(pool.imap_unordered(run_task, tasks),
                                     total=len(tasks),
                                     desc="GP Experiments")):
            results.append(res)
            i = start_idx + rel_i

            # Teilergebnisse alle 50 Tasks oder am Ende
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                df_part = pd.DataFrame(results)
                df_part.to_csv(partial_csv, index=False)

                os.chdir(os.path.join('..', "/data"))
                msg = f"Partial after {i+1} tasks @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
                commit_and_push_data(PARTIAL_NAME, msg)
                os.chdir(os.path.join('..', '/SLIM'))

    # 4) Finale Ergebnisse speichern und pushen
    df_full = pd.DataFrame(results)
    df_full.to_csv(final_csv, index=False)

    os.chdir(os.path.join('..', "/data"))
    commit_and_push_data(FINAL_NAME,
                         f"Final results @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    print("Fertig – alle Ergebnisse in /data gespeichert und gepusht.")
