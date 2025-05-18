import os
import platform # To check OS for affinity settings

def configure_single_core_execution():
    """
    Attempts to configure the environment for single-core, single-threaded execution.

    This function does the following:
    1. Sets environment variables to instruct numerical libraries (used by NumPy, etc.)
       to use only one thread. This is the primary way to control their parallelism.
    2. Optionally, attempts to set CPU affinity to bind the script's process to a
       single CPU core (core 0 by default). This is a more forceful way to ensure
       only one core is used, but its availability and method vary by OS.
    3. Advises checking the specific library ('slim_gsgp_lib_np') for any internal
       parallelism settings it might have.
    """
    print("Configuring for single-core, single-threaded execution...")

    # Step 1: Set environment variables for threading control in numerical libraries
    # These should be set BEFORE importing numpy or other libraries that might use them.
    # Common variables for OpenMP, MKL, OpenBLAS, etc.
    env_vars_to_set = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", # For macOS Accelerate framework
        "NUMEXPR_NUM_THREADS": "1",   # For Numexpr library
        # "CUDA_VISIBLE_DEVICES": ""    # Uncomment if you want to ensure no GPU is used
                                      # by any library that might try to offload computations.
    }

    for var_name, value in env_vars_to_set.items():
        os.environ[var_name] = value
        print(f"  Environment variable set: {var_name}={value}")

    # Step 2: Attempt to set CPU affinity (bind process to a specific core)
    # This is OS-dependent. We'll try for Linux and provide advice for Windows.
    try:
        # hasattr check is good practice for cross-platform compatibility
        if hasattr(os, 'sched_setaffinity'):
            # This function is available on Linux and some other Unix-like systems.
            pid = os.getpid()  # Get current process ID
            # Set affinity to the first CPU core (core 0).
            # The second argument is a set of CPU indices. {0} means only CPU 0.
            os.sched_setaffinity(pid, {0})
            print(f"  Successfully set CPU affinity for PID {pid} to core 0 (Linux/Unix).")
        elif platform.system() == "Windows":
            # Setting affinity from within a Python script on Windows without external
            # libraries like pywin32 or psutil is non-trivial.
            # The most straightforward way is to launch the script with affinity set.
            print("  CPU Affinity on Windows: For strict single-core execution,")
            print("  it's best to launch this script using the 'start' command:")
            print("  > start /affinity 1 python your_script_name.py")
            print("  (This binds the process to the first CPU core, mask '1').")
            # For advanced users, ctypes can be used, but it's more complex:
            # import ctypes
            # kernel32 = ctypes.WinDLL('kernel32')
            # handle = kernel32.GetCurrentProcess()
            # # Affinity mask for the first core is 1 (binary ...0001).
            # # Ensure this mask is valid for your system.
            # if kernel32.SetProcessAffinityMask(handle, 1):
            #    print(f"  (ctypes) Successfully set CPU affinity to core 0 on Windows.")
            # else:
            #    print(f"  (ctypes) Failed to set CPU affinity on Windows (Error: {kernel32.GetLastError()}).")
        else:
            print(f"  CPU affinity setting via 'os.sched_setaffinity' not available on this OS ({platform.system()}).")
            print(f"  CPU affinity via 'taskset' (Linux) or 'start /affinity' (Windows) command line may be an alternative.")

    except Exception as e:
        print(f"  Warning: Could not set CPU affinity: {e}")
        print(f"  Ensure you have necessary permissions if using 'os.sched_setaffinity'.")

    print("-" * 50)
    print("IMPORTANT: Single-core configuration relies on these settings.")
    print("Please verify that 'slim_gsgp_lib_np' does not have its own internal")
    print("multi-threading or multiprocessing mechanisms that override these settings.")
    print("Check its documentation for parameters like 'n_jobs=1' or similar.")
    print("-" * 50)


# --- Call this configuration function at the VERY START of your script ---
configure_single_core_execution()


# --- Now, proceed with your other imports and the rest of your script ---
# Standard library imports
import time
# import os # Already imported above
import pickle

# Third-party imports
import numpy as np # NumPy will now pick up the environment variables
# import matplotlib.pyplot as plt

# Local library imports (from slim_gsgp_lib_np)
from slim_gsgp_lib_np.utils.utils import train_test_split
from slim_gsgp_lib_np.main_multi_slim import multi_slim
from slim_gsgp_lib_np.datasets.synthetic_datasets import load_synthetic10
from slim_gsgp_lib_np.config.multi_slim_config import GPParameters

# ... (rest of your previously refactored script, including
#      DATASET_LOADER, GP_PARAMS_CONFIG, MULTI_SLIM_SEED, etc.
#      and the run_gp_benchmark() function and the if __name__ == "__main__": block)
# --- Configuration Constants ---
# Dataset Configuration
DATASET_LOADER = load_synthetic10
DATASET_NAME_STR = 'Synthetic10' # For logging and potential output naming
NUM_SAMPLES = 3000
TEST_SET_FRACTION = 0.3
BASE_RANDOM_SEED = 0 # Base seed for reproducibility

# Genetic Programming (GP) Engine Parameters
GP_PARAMS_CONFIG = GPParameters(
    pop_size=500,
    n_iter=100,
    max_depth=7,
    init_depth=3,
    particularity_pressure=5,
    selector='dalex',
    p_xo=0.8,
    down_sampling=1,
    dalex_size_prob=1
)

# MultiSLIM Framework Parameters
MULTI_SLIM_SEED = BASE_RANDOM_SEED + 10
MULTI_SLIM_POP_SIZE = 100
MULTI_SLIM_ITERATIONS = 100
MULTI_SLIM_XO_PROB = 0.4
MULTI_SLIM_PROB_TERMINAL = 0.7
MULTI_SLIM_PROB_SPECIALIST = 0.7
MULTI_SLIM_MAX_DEPTH = 4
MULTI_SLIM_DEPTH_CONDITION = 5
MULTI_SLIM_USE_TEST_ELITE = True
MULTI_SLIM_TIMEOUT_SECONDS = 200
MULTI_SLIM_PARTICULARITY_PRESSURE = 2


def run_gp_benchmark():
    """
    Runs the genetic programming benchmark using multi_slim.
    This function loads data, splits it, configures parameters,
    runs the GP process, and prints the time taken.
    """
    print(f"Starting benchmark for dataset: {DATASET_NAME_STR}...")

    features_all, target_all, _, _ = DATASET_LOADER(n=NUM_SAMPLES)

    train_idx, test_idx = train_test_split(features_all, target_all,
                                           p_test=TEST_SET_FRACTION,
                                           seed=BASE_RANDOM_SEED,
                                           indices_only=True)
    features_train, target_train = features_all[train_idx], target_all[train_idx]
    features_test, target_test = features_all[test_idx], target_all[test_idx]

    print(f"Data prepared: Train shapes ({features_train.shape}, {target_train.shape}), Test shapes ({features_test.shape}, {target_test.shape})")

    print(f"Executing multi_slim with '{GP_PARAMS_CONFIG.selector}' selector for GP engine...")
    time_start = time.time()

    _ensemble_models, _multi_pop_details, _spec_pop_details = multi_slim(
        X_train=features_train, y_train=target_train,
        X_test=features_test, y_test=target_test,
        dataset_name=DATASET_NAME_STR,
        gp_version='gp',
        params_gp=GP_PARAMS_CONFIG,
        selector=GP_PARAMS_CONFIG.selector,
        full_return=True,
        seed=MULTI_SLIM_SEED,
        verbose=0,
        pop_size=MULTI_SLIM_POP_SIZE,
        n_iter=MULTI_SLIM_ITERATIONS,
        p_xo=MULTI_SLIM_XO_PROB,
        prob_terminal=MULTI_SLIM_PROB_TERMINAL,
        prob_specialist=MULTI_SLIM_PROB_SPECIALIST,
        max_depth=MULTI_SLIM_MAX_DEPTH,
        depth_condition=MULTI_SLIM_DEPTH_CONDITION,
        test_elite=MULTI_SLIM_USE_TEST_ELITE,
        timeout=MULTI_SLIM_TIMEOUT_SECONDS,
        particularity_pressure=MULTI_SLIM_PARTICULARITY_PRESSURE
    )

    time_end = time.time()
    time_elapsed_seconds = time_end - time_start

    print(f"Benchmarking for {DATASET_NAME_STR} completed.")
    print(f"Total time taken: {time_elapsed_seconds:.2f} seconds.")


if __name__ == "__main__":
    print('Benchmarking GP (Refactored Script - Configured for Single Core)')
    # The configure_single_core_execution() function is called globally at the top.
    run_gp_benchmark()