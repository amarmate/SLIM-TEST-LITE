import os
import time
import psutil
import multiprocessing
from joblib import Parallel, delayed
from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
import numpy as np

# Force single-threaded numeric libs
os.environ.update({k: '1' for k in [
    "OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"
]})

try:
    profile
except NameError:
    def profile(func):
        return func


def dataset3(n=500, seed=0, noise=0):
    np.random.seed(seed)
    x = np.random.uniform(-3,3,(n,2))
    x[:,1] *= 0.1
    def f3(xi): return xi[0]**2 if xi[0]<-1 else xi[0]**4-3*xi[0]**3+2*xi[0]**2-4
    y = np.array([f3(xi) for xi in x])
    if noise>0:
        y += np.random.normal(0,(noise/100)*np.std(y),size=n)
    return train_test_split(x,y,p_test=0.2,seed=0)

Xtr, Xte, ytr, yte = dataset3(n=2000, seed=0, noise=0)

def measure(seed, label=None):
    proc = psutil.Process()
    mem0 = proc.memory_info().rss / 1024**2
    t0 = time.time()

    res = gp(
        X_train=Xtr, y_train=ytr, test_elite=False,
        dataset_name=f"run_{seed}",
        pop_size=100, n_iter=200, selector='dalex_fast_rand',
        max_depth=8, init_depth=2, p_xo=0.8,
        prob_const=0.2, prob_terminal=0.7, tournament_size=2, 
        particularity_pressure=10, seed=seed, 
        full_return=False, n_jobs=1, verbose=True,
        log_level=0, tree_functions=['add','multiply','subtract','AQ'],
        it_tolerance=20000, dalex_n_cases=5, down_sampling=1
    )

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
    sc = measure(seed=10, label="single")
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