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

def dataset3(n=500, seed=0, noise=0):
    np.random.seed(seed)
    x = np.random.uniform(-3,3,(n,2))
    x[:,1] *= 0.1
    def f3(xi): return xi[0]**2 if xi[0]<-1 else xi[0]**4-3*xi[0]**3+2*xi[0]**2-4
    y = np.array([f3(xi) for xi in x])
    if noise>0:
        y += np.random.normal(0,(noise/100)*np.std(y),size=n)
    return train_test_split(x,y,p_test=0.2,seed=0)

def measure(run_id):
    proc = psutil.Process()
    mem0 = proc.memory_info().rss/1024**2
    t0 = time.time()
    Xtr, Xte, ytr, yte = dataset3(n=1000)
    gp(X_train=Xtr, y_train=ytr, test_elite=False,
       dataset_name=f"run{run_id}", pop_size=100, n_iter=500,
       selector='dalex_fast_rand', max_depth=9, init_depth=2,
       p_xo=0.8, prob_const=0.2, prob_terminal=0.7,
       particularity_pressure=10, seed=run_id,
       full_return=False, n_jobs=1, verbose=False, log_level=0,
       tree_functions=['add','multiply','subtract','AQ'], it_tolerance=20000,
       dalex_n_cases=5, down_sampling=1)
    dt = time.time() - t0
    mem1 = proc.memory_info().rss/1024**2
    return {"run": run_id, "time_s": dt, "mem_start_MB": mem0, "mem_end_MB": mem1}

if __name__ == "__main__":
    # Single-Core
    sc = measure("single")
    print("Single-Core:", sc)

    # Multiprocessing
    nproc = max(1, psutil.cpu_count(logical=True)-1)
    with multiprocessing.Pool(nproc) as pool:
        mp = pool.map(measure, range(nproc))
    print("Multiprocessing:", mp)

    # Joblib
    jb = Parallel(n_jobs=nproc)(
        delayed(measure)(i) for i in range(nproc)
    )
    print("Joblib Parallel:", jb)
