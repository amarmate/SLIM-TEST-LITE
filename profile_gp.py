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

Xtr, Xte, ytr, yte = dataset3(n=2000, seed=0, noise=0)

def measure(seed, label=None):
    proc = psutil.Process()
    mem0 = proc.memory_info().rss / 1024**2
    t0 = time.time()

    # … Daten generieren und train/test split …

    # Verwende seed als Integer, nicht label
    res = gp(
        X_train=Xtr, y_train=ytr, test_elite=False,
        dataset_name=f"run_{seed}",
        pop_size=100, n_iter=1000, selector='dalex_fast_rand',
        max_depth=9, init_depth=2, p_xo=0.8,
        prob_const=0.2, prob_terminal=0.7,
        particularity_pressure=10, seed=seed,  # <-- ganzzahliger seed
        full_return=False, n_jobs=1, verbose=False,
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
    sc = measure(seed=0, label="single")
    print("Single-Core:", sc)

    # Multiprocessing: seeds 1,2,...
    with multiprocessing.Pool(nproc) as pool:
        mp_188 = pool.starmap(measure, [(0, f"proc_{i+1}") for i in range(188)])

    with multiprocessing.Pool(nproc) as pool:
        mp_160 = pool.starmap(measure, [(0, f"proc_{i+1}") for i in range(160)])

    # Joblib: dieselben seeds
    from joblib import Parallel, delayed
    jb_188 = Parallel(n_jobs=nproc)(
        delayed(measure)(0, f"joblib_{i+1}") for i in range(160)
    )

    jb_120 = Parallel(n_jobs=nproc)(
        delayed(measure)(0, f"joblib_{i+1}") for i in range(120)
    )

    import pickle
    pickle.dump(sc, open("sc.pickle", "wb"))
    pickle.dump(mp_188, open("mp_188.pickle", "wb"))
    pickle.dump(mp_160, open("mp_160.pickle", "wb"))
    pickle.dump(jb_188, open("jb_188.pickle", "wb"))
    pickle.dump(jb_120, open("jb_120.pickle", "wb"))

    # Now zip all the results
    import zipfile
    with zipfile.ZipFile("results.zip", "w") as zf:
        zf.write("sc.pickle")
        zf.write("mp_188.pickle")
        zf.write("mp_160.pickle")
        zf.write("jb_188.pickle")
        zf.write("jb_120.pickle")
        zf.close()
    # Clean up
    os.remove("sc.pickle")
    os.remove("mp_188.pickle")
    os.remove("mp_160.pickle")
    os.remove("jb_188.pickle")
    os.remove("jb_120.pickle")
    

