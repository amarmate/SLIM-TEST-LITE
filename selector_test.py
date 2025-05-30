from joblib import Parallel, delayed
from tqdm.auto import tqdm
import multiprocessing
import time
import pickle
import os
import numpy as np
import itertools
import argparse

from slim_gsgp_lib_np.utils.callbacks import LogSpecialist
from slim_gsgp_lib_np.main_gp import gp
from slim_gsgp_lib_np.datasets.synthetic_datasets import (
    load_synthetic2, load_synthetic3, load_synthetic4,
    load_synthetic5, load_synthetic6, load_synthetic7,
    load_synthetic8, load_synthetic9, load_synthetic10,
)

BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
]
for env_var in BLAS_THREAD_ENV_VARS:
    os.environ[env_var] = '1'


def _single_run(
        features: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        num_iterations: int,
        population_size: int,
        selector_method: str,
        particularity_pressure_param: float,
        dataset_name_str: str,
        run_seed: int,
        dalex_size_probability: float
    ) -> dict:
    start_time = time.time()
    callback_logger = LogSpecialist(features, target, mask)
    pp_value_for_gp = particularity_pressure_param if selector_method != 'tournament' else None

    gp(
        X_train=features,
        y_train=target,
        test_elite=False,
        callbacks=[callback_logger],
        full_return=True,
        n_iter=num_iterations,
        pop_size=population_size,
        seed=run_seed,
        max_depth=7,
        init_depth=2,
        selector=selector_method,
        tournament_size=2,
        particularity_pressure=pp_value_for_gp,
        verbose=0,
        dataset_name=dataset_name_str,
        dalex_size_prob=dalex_size_probability
    )

    runtime_seconds = time.time() - start_time
    # final_rmse = callback_logger.log_rmse[-1]
    # final_size = callback_logger.log_size[-1]
    # final_rmse_out_of_sample = callback_logger.log_rmse_out[-1]

    # rmse_evolution = np.array(callback_logger.log_rmse)
    # min_rmse_indices = np.argmin(rmse_evolution, axis=0)
    # min_rmse_values = np.min(rmse_evolution, axis=0)
    # max_rmse_values = np.max(rmse_evolution, axis=0)
    # convergence_threshold = 0.95 * min_rmse_values + 0.05 * max_rmse_values

    # convergence_iterations = []
    # for i in range(rmse_evolution.shape[1]):
    #     iterations_below_threshold = np.where(rmse_evolution[:, i] < convergence_threshold[i])[0]
    #     convergence_iterations.append(iterations_below_threshold)

    return {
        'dataset': dataset_name_str,
        'n_iter': num_iterations,
        'pop_size': population_size,
        'pp': pp_value_for_gp,
        'dp': dalex_size_probability,
        'seed': run_seed,
        'selector_method_used': selector_method,
        # 'result': final_rmse,
        # 'result_out': final_rmse_out_of_sample,
        # 'min_idx': min_rmse_indices,
        # 'results_conv': convergence_iterations,
        # 'size': final_size,
        'results_rmse' : callback_logger.log_rmse,
        'results_out' : callback_logger.log_rmse_out,
        'results_size' : callback_logger.log_size,
        'duration': runtime_seconds,
    }


def collect_logs_experiment(
        dataset_loader_func,
        dataset_name: str,
        *,
        num_iterations: int,
        population_size: int,
        selector_method: str,
        particularity_pressure_param: float = None,
        num_runs: int = 30,
        base_random_seed: int = 5,
        num_parallel_jobs: int = -1,
        dalex_size_prob_param: float = 0.5
    ) -> list:
    features, target, _, mask = dataset_loader_func()

    print(
        f"Starting experiment group: {dataset_name} | Selector={selector_method} | Iter={num_iterations} | "
        f"Pop={population_size} | PP={particularity_pressure_param} | "
        f"DP={dalex_size_prob_param} | Runs={num_runs}"
    )

    run_seeds = [base_random_seed + i for i in range(num_runs)]
    tasks = (
        delayed(_single_run)(
            features, target, mask, num_iterations, population_size,
            selector_method, particularity_pressure_param, dataset_name, seed, dalex_size_prob_param
        )
        for seed in run_seeds
    )
    experiment_results = Parallel(n_jobs=num_parallel_jobs)(tasks)
    return experiment_results


def aggregate_run_logs(individual_run_logs: list, current_selector_method: str) -> dict:
    if not individual_run_logs:
        return {}

    aggregated_results = {
        'dataset': individual_run_logs[0]['dataset'],
        'selector': individual_run_logs[0].get('selector_method_used', current_selector_method),
        'n_iter': individual_run_logs[0]['n_iter'],
        'pop_size': individual_run_logs[0]['pop_size'],
        'pp': individual_run_logs[0]['pp'],
        'dp': individual_run_logs[0]['dp'],
        'result_rmse': [],
        'result_out': [],
        'result_size': [],
        'duration': [],
    }

    for run_log in individual_run_logs:
        aggregated_results['result_rmse'].append(run_log['results_rmse'])
        aggregated_results['result_out'].append(run_log['results_out'])
        aggregated_results['result_size'].append(run_log['results_size'])
        aggregated_results['duration'].append(run_log['duration'])

    aggregated_results['result_rmse'] = np.array(aggregated_results['result_rmse'])
    aggregated_results['result_out'] = np.array(aggregated_results['result_out'])
    aggregated_results['result_size'] = np.array(aggregated_results['result_size'])
    aggregated_results['duration'] = np.array(aggregated_results['duration'])
    
    return aggregated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
    parser.add_argument(
        '--workers', type=int, default=1,
        help="Number of parallel workers to use for the experiments."
    )
    args = parser.parse_args()

    datasets_to_process = [
        (load_synthetic2, 'synthetic2'), (load_synthetic3, 'synthetic3'),
        (load_synthetic4, 'synthetic4'), (load_synthetic5, 'synthetic5'),
        (load_synthetic6, 'synthetic6'), (load_synthetic7, 'synthetic7'),
        (load_synthetic8, 'synthetic8'), (load_synthetic9, 'synthetic9'),
        (load_synthetic10, 'synthetic10'),
    ]
    iteration_population_configs = [(400, 200), (100, 500), (40, 1000), (20, 2000)]
    particularity_pressure_configs = [3, 4, 5, 7, 10, 12, 15, 17, 20, 25]
    gp_selector_method = 'dalex_size'
    dalex_size_probability_configs = [0.4, 0.6, 0.8, 1.0]

    num_available_cores = multiprocessing.cpu_count()
    cores_for_parallelism = max(1, num_available_cores - 1)
    cores_to_use = min(cores_for_parallelism, args.workers)
    print(f"Using {cores_to_use} cores for parallel processing.")

    output_directory = 'logs_experiment_dalex_size_single_loop'
    os.makedirs(output_directory, exist_ok=True)

    all_experiment_params = list(itertools.product(
        datasets_to_process,
        iteration_population_configs,
        particularity_pressure_configs,
        dalex_size_probability_configs
    ))

    for experiment_config in tqdm(all_experiment_params, desc="Total Experiments"):
        (loader_func, dataset_name_str), (num_iter, pop_s), pp_config, dp_config = experiment_config

        all_run_logs = collect_logs_experiment(
            dataset_loader_func=loader_func,
            dataset_name=dataset_name_str,
            num_iterations=num_iter,
            population_size=pop_s,
            selector_method=gp_selector_method,
            particularity_pressure_param=pp_config,
            dalex_size_prob_param=dp_config,
            num_runs=30,
            base_random_seed=5,
            num_parallel_jobs=cores_to_use
        )

        if all_run_logs:
            aggregated_experiment_logs = aggregate_run_logs(all_run_logs, gp_selector_method)
            experiment_identifier = (
                f"{dataset_name_str}_{gp_selector_method}_it{num_iter}_pop{pop_s}"
                f"_pp{pp_config}_dp{dp_config}_1605"
            )
            output_filename = os.path.join(output_directory, experiment_identifier + ".pkl")
            with open(output_filename, 'wb') as f_out:
                pickle.dump(aggregated_experiment_logs, f_out)
        else:
            print(f"Warning: No logs generated for config: {experiment_config}, skipping aggregation and saving.")


    print(f"\nFinished all experiments.")
    print(f"Saved experiments results to '{output_directory}'.")
    print(f"Total files expected (if all runs succeeded): {len(all_experiment_params)}")
    print(f"Total files actually generated: {len(os.listdir(output_directory))}")