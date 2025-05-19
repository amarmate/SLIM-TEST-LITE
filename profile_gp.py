from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_gp import gp
import numpy as np
import time
import os
import psutil # For measuring CPU and RAM
import sys    # For checking platform for resource module units
import multiprocessing # For running experiments in parallel

# Attempt to import the resource module for peak memory usage
try:
    import resource
    RESOURCE_MODULE_AVAILABLE = True
except ImportError:
    RESOURCE_MODULE_AVAILABLE = False

# Set environment variables to ensure each GP process is single-threaded internally
# This is important when running multiple GP instances in parallel via multiprocessing
os.environ.update({
    k: '1' for k in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
    ]
})


def dataset3(n=500, seed=0, noise=0):
    """
    Generates a synthetic dataset.
    The dataset features a piecewise function with a threshold.
    It includes an option for adding noise and may have a slight class imbalance.
    """
    def f3(x_sample):
        if x_sample[0] < -1:
            return x_sample[0]**2
        else:
            return x_sample[0]**4 - 3*x_sample[0]**3 + 2*x_sample[0]**2 - 4

    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 2))
    # Making the second feature less impactful or scaled differently
    x[:, 1] = x[:, 1] * 0.1 
    
    # For information purposes, printing class distribution based on the condition x[0] < -1
    # print(f'Class 1 (x[0] < -1) has {np.sum(x[:, 0] < -1)} samples, and Class 2 (x[0] >= -1) has {np.sum(x[:, 0] >= -1)} samples.')
    
    y_clean = np.array([f3(xi) for xi in x])
    
    if noise > 0:
        std_dev = np.std(y_clean)
        y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    else:
        y_noisy = y_clean
        
    # Mask generation, not explicitly used in the provided run_gp but part of dataset definition
    mask_class1 = x[:, 0] < -1
    mask = [mask_class1, np.logical_not(mask_class1)]
    
    return x, y_noisy, mask

FUNCTIONS = ['add', 'multiply', 'subtract', 'AQ']

def run_gp_experiment(experiment_id_info="Single Core"):
    """
    Runs a single genetic programming experiment and measures its resource usage.
    This function encapsulates the core GP execution and resource monitoring.
    It prints CPU times (user and system) and memory usage (RSS and VMS)
    at the start and end of the GP run. It also reports peak memory usage if possible.
    """
    print(f"\n--- Starting GP Experiment ({experiment_id_info}) ---")
    process = psutil.Process(os.getpid())

    mem_info_start = process.memory_info()
    cpu_times_start = process.cpu_times()
    print(f"({experiment_id_info}) Initial RAM: RSS={mem_info_start.rss / 1024**2:.2f} MB, VMS={mem_info_start.vms / 1024**2:.2f} MB")
    print(f"({experiment_id_info}) Initial CPU: User={cpu_times_start.user:.2f}s, System={cpu_times_start.system:.2f}s")

    X, y, _ = dataset3(n=1000, seed=0, noise=0) # Using fixed seed for consistent datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed=0)


    wall_time_start = time.time()
    # The gp function is the core of the experiment.
    # Parameters are set for a typical run.
    res = gp(X_train=X_train, y_train=y_train, test_elite=False, dataset_name=f'test_{experiment_id_info.replace(" ", "_")}',
             pop_size=int(100), n_iter=int(2000), selector='dalex_fast_rand',
             max_depth=9, init_depth=2, p_xo=0.8, prob_const=0.2,
             prob_terminal=0.7, particularity_pressure=10, seed=0, # Fixed seed for GP
             full_return=True, n_jobs=1, verbose=False, log_level=0, # n_jobs=1 is crucial here
             tree_functions=FUNCTIONS, it_tolerance=20000,
             dalex_n_cases=5, down_sampling=1)
    wall_time_end = time.time()
    
    wall_time_taken = wall_time_end - wall_time_start
    print(f"({experiment_id_info}) Wall clock time for GP: {wall_time_taken:.2f}s")

    mem_info_end = process.memory_info()
    cpu_times_end = process.cpu_times()
    print(f"({experiment_id_info}) Final RAM: RSS={mem_info_end.rss / 1024**2:.2f} MB, VMS={mem_info_end.vms / 1024**2:.2f} MB")
    print(f"({experiment_id_info}) Final CPU: User={cpu_times_end.user:.2f}s, System={cpu_times_end.system:.2f}s")

    cpu_time_user_spent = cpu_times_end.user - cpu_times_start.user
    cpu_time_system_spent = cpu_times_end.system - cpu_times_start.system
    print(f"({experiment_id_info}) Total CPU time by process: User={cpu_time_user_spent:.2f}s, System={cpu_time_system_spent:.2f}s")

    if RESOURCE_MODULE_AVAILABLE:
        peak_mem_rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin': # macOS reports in Bytes
            peak_mem_kb = peak_mem_rusage / 1024
        else: # Linux and others typically report in KB
            peak_mem_kb = peak_mem_rusage
        print(f"({experiment_id_info}) Peak memory (ru_maxrss): {peak_mem_kb:.2f} KB ({peak_mem_kb / 1024:.2f} MB)")
    else:
        print(f"({experiment_id_info}) Peak memory via 'resource' module not available. RSS can be an indicator.")
    print(f"--- Finished GP Experiment ({experiment_id_info}) ---")
    # If 'res' contained results to be returned, this function would return them.
    # For now, it only prints information.

def run_gp_for_pool(run_index):
    """
    Wrapper function to call run_gp_experiment with an identifier for parallel runs.
    This function is designed to be used with multiprocessing.Pool.map.
    It assigns a unique ID to each parallel experiment for easier tracking in logs.
    """
    run_gp_experiment(experiment_id_info=f"Parallel Run {run_index + 1}")


if __name__ == "__main__":
    # --- 1. Single Core Experiment ---
    print(">>> Running Single Core GP Experiment <<<")
    run_gp_experiment(experiment_id_info="Single Core Baseline")

    # --- 2. Parallel Experiments (Available Threads - 2) ---
    print("\n>>> Preparing for Parallel GP Experiments <<<")
    
    available_threads = psutil.cpu_count(logical=True)
    num_parallel_experiments = available_threads - 2
    
    print(f"Available logical CPU threads: {available_threads}")
    print(f"Number of parallel experiments to run: {num_parallel_experiments} (available_threads - 2)")

    if num_parallel_experiments > 0:
        print(f"\n>>> Starting {num_parallel_experiments} Parallel GP Experiments <<<")
        # Ensure the main process also adheres to single-thread for its operations if any NumPy/etc. are used here
        # though for Pool, this is less critical for the parent itself.
        
        # Using multiprocessing.Pool to run experiments in parallel
        # Each call to run_gp_for_pool will execute in a separate process.
        with multiprocessing.Pool(processes=num_parallel_experiments) as pool:
            # pool.map will call run_gp_for_pool for each item in range(num_parallel_experiments)
            # run_gp_for_pool(0), run_gp_for_pool(1), ..., run_gp_for_pool(num_parallel_experiments-1)
            pool.map(run_gp_for_pool, range(num_parallel_experiments))
        
        print("\n>>> All Parallel GP Experiments Finished <<<")
    elif num_parallel_experiments == 0:
        print("Not enough threads to run parallel experiments (available_threads - 2 = 0).")
        print("This typically means you have 2 logical CPU threads.")
    else: # num_parallel_experiments < 0
        print("Not enough threads to run parallel experiments (available_threads - 2 < 0).")
        print("This typically means you have 1 logical CPU thread.")