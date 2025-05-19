from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
import numpy as np
import time
import os
import psutil # For measuring CPU and RAM
import sys    # For checking platform for resource module units

# Attempt to import the resource module for peak memory usage
try:
    import resource
    RESOURCE_MODULE_AVAILABLE = True
except ImportError:
    RESOURCE_MODULE_AVAILABLE = False

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

FUNCTIONS = ['add', 'multiply', 'subtract', 'AQ']

def run_gp():
    """
    This function runs the genetic programming algorithm and measures
    its CPU and RAM usage.
    It records CPU times (user and system) and memory usage (RSS and VMS)
    at the beginning and end of the core GP execution.
    It also attempts to report peak memory usage using the 'resource' module
    if available.
    """
    # --- Resource Usage Measurement Setup ---
    process = psutil.Process(os.getpid())

    # Initial memory and CPU times
    mem_info_start = process.memory_info()
    cpu_times_start = process.cpu_times()
    print(f"Initial RAM usage: RSS={mem_info_start.rss / 1024**2:.2f} MB, VMS={mem_info_start.vms / 1024**2:.2f} MB")
    print(f"Initial CPU times: User={cpu_times_start.user:.2f}s, System={cpu_times_start.system:.2f}s")

    # --- Original Script Logic ---
    X, y, _ = dataset3(n=1000, seed=0, noise=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2)

    start_time = time.time() # Wall clock time
    res = gp(X_train=X_train, y_train=y_train, test_elite=False, dataset_name='test',
             pop_size=int(100), n_iter=int(250), selector='dalex_fast_rand',
             max_depth=9, init_depth=2, p_xo=0.8, prob_const=0.2,
             prob_terminal=0.7, particularity_pressure=10, seed=0,
             full_return=True, n_jobs=1, verbose=True, log_level=0,
             tree_functions=FUNCTIONS, it_tolerance=20000,
             dalex_n_cases=5, down_sampling=1)

    end_time = time.time() # Wall clock time
    wall_time_taken = end_time - start_time
    print(f"Wall clock time taken for GP: {wall_time_taken:.2f}s")

    # --- Resource Usage Measurement Reporting ---
    # Final memory and CPU times
    mem_info_end = process.memory_info()
    cpu_times_end = process.cpu_times()

    print(f"Final RAM usage: RSS={mem_info_end.rss / 1024**2:.2f} MB, VMS={mem_info_end.vms / 1024**2:.2f} MB")
    print(f"Final CPU times: User={cpu_times_end.user:.2f}s, System={cpu_times_end.system:.2f}s")

    # Calculate and print CPU time spent by the process
    cpu_time_user_spent = cpu_times_end.user - cpu_times_start.user
    cpu_time_system_spent = cpu_times_end.system - cpu_times_start.system
    print(f"Total CPU time spent by process: User={cpu_time_user_spent:.2f}s, System={cpu_time_system_spent:.2f}s")

    # Report peak memory usage if resource module is available
    if RESOURCE_MODULE_AVAILABLE:
        peak_mem_rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss unit varies by platform: KB on Linux, Bytes on macOS
        if sys.platform == 'darwin': # macOS reports in Bytes
            peak_mem_kb = peak_mem_rusage / 1024
        else: # Linux and others typically report in KB
            peak_mem_kb = peak_mem_rusage
        print(f"Peak memory usage (ru_maxrss): {peak_mem_kb:.2f} KB ({peak_mem_kb / 1024:.2f} MB)")
    else:
        print("Peak memory usage via 'resource' module is not available on this platform.")
        print("The 'Final RAM usage' (RSS) can be an indicator of high memory usage.")


if __name__ == "__main__":
    run_gp()