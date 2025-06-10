import os
import time
import psutil
import multiprocessing
from joblib import Parallel, delayed
from slim_gsgp_lib_np.utils.utils import train_test_split
from functions.metrics_test import rmse, r_squared
from slim_gsgp_lib_np.datasets.data_loader import load_diabetes
from slim_gsgp_lib_np.main_gp import gp
import numpy as np

# Force single-threaded numeric libs
os.environ.update({k: '1' for k in [
    "OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"
]})

# try:
#     profile
# except NameError:
#     def profile(func):
#         return func

X,y = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(X, y, p_test=0.2, seed=0)

def measure(seed, label=None):
    proc = psutil.Process()
    mem0 = proc.memory_info().rss / 1024**2
    t0 = time.time()

    res = gp(
        X_train=Xtr, y_train=ytr, test_elite=True, X_test=Xte, y_test=yte,
        dataset_name=f"run_{seed}",
        pop_size=100, n_iter=1000, selector='dalex',
        max_depth=7, init_depth=2, p_xo=0.8,
        prob_const=0.2, prob_terminal=0.7, tournament_size=2, 
        particularity_pressure=10, seed=seed, 
        full_return=True, n_jobs=1, verbose=True,
        log_level=0, tree_functions=['add','multiply', 'sq'
                                     # 'subtract'
                                     # , 'divide', 'AQ'
                                     ],
        it_tolerance=20000, dalex_n_cases=5, down_sampling=1
    )
    pred = res.elite.predict(Xte)
    print('RMSE:', rmse(yte, pred))
    print('R^2:', r_squared(yte, pred))

    dt = time.time() - t0
    mem1 = proc.memory_info().rss / 1024**2
    return {
        "label": label or f"run_{seed}",
        "seed": seed,
        "time_s": dt,
        "mem_start_MB": mem0,
        "mem_end_MB": mem1
    }

if __name__ == "__main__":
    # Single-Core: seed = 0
    sc = measure(seed=0, label="single")
    # print("Single-Core:", sc)

    # from joblib import Parallel, delayed
    # j_data = {}
    # for jobs in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    #     j = Parallel(n_jobs=jobs)(
    #         delayed(measure)(0, f"joblib_{i+1}") for i in range(jobs)
    #     )
    #     j_data[jobs] = j

    # # Print the mean time_s for each job 
    # for jobs, data in j_data.items():
    #     mean_time = np.mean([d["time_s"] for d in data])
    #     print(f"Joblib {jobs} jobs: {mean_time:.2f} seconds -> {mean_time/jobs} seconds/job")