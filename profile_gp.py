from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
import numpy as np
import time
import os 

os.environ.update({
    k: '1' for k in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
    ]})


def dataset3(n=500, seed=0, noise=0):
    """
    Synthetic dataset with a single threshold-based piecewise function and a slight class imbalance
    The functions are harder
    """
    def f3(x): 
        if x[0] < -1: 
            return x[0]**2 
        else: 
            return x[0]**4 - 3*x[0]**3 + 2*x[0]**2 - 4
    
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class 1 has', np.sum(x[:, 0] < -1), 'samples, and class 2 has', np.sum(x[:, 0] >= -1), 'samples')
    y_clean = np.array([f3(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask = np.where(x[:, 0] < -1, True, False)
    mask = [mask, np.logical_not(mask)]
    return x, y_noisy, mask


def run_gp():
    start = time.time()
    X, y, _ = dataset3(n=2000, seed=0, noise=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2)

    elite, gp_pop = gp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_elite=True,
                      full_return=True, dataset_name='test', selector='dalex_fast', n_iter=200, pop_size=100, 
                      verbose=1, max_depth=8, init_depth=3, p_xo=0.8, seed=0, down_sampling=1, particularity_pressure=10,
    )
    print('Time taken:', time.time() - start)


if __name__ == "__main__":
    run_gp()
