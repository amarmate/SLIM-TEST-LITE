import pickle
import os
import numpy as np
import pandas as pd
import platform
import psutil
import sympy as sp
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mlflow
import re
from pathlib import Path

def pf_rmse_comp(points):
    pareto = []
    for i, (rmse1, comp1) in enumerate(points):
        dominated = False
        for j, (rmse2, comp2) in enumerate(points):
            if j != i and (rmse2 <= rmse1 and comp2 <= comp1) and (rmse2 < rmse1 or comp2 < comp1):
                dominated = True
                break
        if not dominated:
            pareto.append((rmse1, comp1))
    # Optionally sort by RMSE or complexity
    pareto.sort()
    return pareto

def pf_rmse_comp_time(points): 
    """
    Generate a Pareto front considering RMSE, complexity, and time.

    Parameters
    ----------
    points : list of tuples (rmse, comp, time)
        A list of individuals from the Pareto front. Each individual is represented as 
        (RMSE, complexity, time)

    Returns
    -------
    list
        A Pareto front containing the selected individuals based on the criteria.
    """

    pareto = []
    for i, (rmse1, comp1, time1) in enumerate(points):
        dominated = False
        for j, (rmse2, comp2, time2) in enumerate(points):
            if j != i and (rmse2 <= rmse1 and comp2 <= comp1 and time2 <= time1) and (rmse2 < rmse1 or comp2 < comp1 or time2 < time1):
                dominated = True
                break
        if not dominated:
            pareto.append((rmse1, comp1, time1))

    pareto.sort(key=lambda x: (x[0], x[1], x[2]))
    return pareto


# -------------------------------------------------------------------- HYPERPARAMETER TUNING -------------------------------------------------------------------------------------------
def save_tuning_results(name: str,
                        split_id: int,
                        df_tr: pd.DataFrame,
                        best_hyperparams: dict,
                        base_dir: str = 'hp_results',
                        prefix: str = 'GP',
                        suffix: int = None):
    """
    Persist hyperparameter tuning results and best‐params per split.

    - Stores/updates per‐split trial log in <base_dir>/<name>/prefix_tr_resultssuffix.pkl
    - Stores/updates best hyperparameters dict in <base_dir>/<name>/prefix_paramssuffix.pkl

    Parameters
    ----------
    name : str
        Experiment name (subfolder under base_dir).
    split_id : int
        Which data‐split these results correspond to.
    df_tr : pd.DataFrame
        All trial results for this split.
    best_hyperparams : dict
        The selected best hyperparameters for this split.
    base_dir : str, default 'hp_results'
        Root folder under which to save results.
    """
    dir_path = os.path.join(base_dir, name)
    os.makedirs(dir_path, exist_ok=True)

    tr_path = os.path.join(dir_path, f'{prefix}_tr{suffix}.pkl')
    if os.path.exists(tr_path):
        # append new trials to existing log
        previous_tr = pd.read_parquet(tr_path)
        combined = pd.concat([previous_tr, df_tr], ignore_index=True)
        combined.to_parquet(tr_path)
    else:
        # first time: write full df
        df_tr.to_parquet(tr_path)

    params_path = os.path.join(dir_path, f'{prefix}_params{suffix}.pkl')
    if os.path.exists(params_path):
        previous_params = pickle.load(open(params_path, 'rb'))
        previous_params[split_id] = best_hyperparams
        pickle.dump(previous_params, open(params_path, 'wb'))
    else:
        # first time: create dict with this split
        new_params = {split_id: best_hyperparams}
        pickle.dump(new_params, open(params_path, 'wb'))



# ------------------------------------------------------------------- TESTING -------------------------------------------------------------------------------------------
def save_test_results(name: str,
                      split_id: int,
                      df_test: pd.DataFrame,
                      base_dir: str = 'test_results',
                      prefix: str = 'GP',
                      suffix: int = None):
    """
    Persist test-phase results per split.

    - Stores/updates per-split test log in <base_dir>/<name>/<prefix>_test_results<suffix>.pkl

    Parameters
    ----------
    name : str
        Experiment name (subfolder under base_dir).
    split_id : int
        Which data-split these results correspond to.
    df_test : pd.DataFrame
        All test results for this split.
    base_dir : str, default 'test_results'
        Root folder under which to save results.
    prefix : str, default 'GP'
        Algorithm prefix for filename.
    suffix : int or str, optional
        Optional suffix to distinguish files (e.g. hyperparam set id).
    """
    # ensure directory exists
    dir_path = os.path.join(base_dir, name)
    os.makedirs(dir_path, exist_ok=True)

    # construct filename
    suf = f"{suffix}" if suffix is not None else ""
    file_path = os.path.join(dir_path, f"{prefix}_test_results{suf}.pkl")

    # append if exists, else write new
    if os.path.exists(file_path):
        prev = pd.read_parquet(file_path)
        combined = pd.concat([prev, df_test], ignore_index=True)
        combined.to_parquet(file_path)
    else:
        df_test.to_parquet(file_path)


def simplify_tuple_expression(expr_tuple, latex=True):
    def parse_expression(expr, variables=None):
        if variables is None:
            variables = {}
        op_map = {
            'add': sp.Add,
            'subtract': lambda a, b: a - b,
            'multiply': sp.Mul,
            'divide': lambda a, b: sp.Mul(a, sp.Pow(b, -1)),
            'sqrt': lambda a: sp.Pow(sp.Abs(a), 0.5),
            'cond': lambda cond, true_val, false_val: sp.Piecewise((true_val, cond > 0), (false_val, True)),
        }

        if isinstance(expr, tuple):
            op = expr[0]
            args = [parse_expression(arg, variables) for arg in expr[1:]]
            if op not in op_map:
                raise ValueError(f"Unknown operator: {op}")
            return op_map[op](*args)

        elif isinstance(expr, str):
            if expr.startswith('x'):
                if expr not in variables:
                    variables[expr] = sp.Symbol(expr)
                return variables[expr]
            elif expr.startswith('constant__') or expr.startswith('constant_'):
                value = expr.replace('constant__', '-').replace('constant_', '')
                return sp.Float(value)
            else:
                if expr not in variables:
                    variables[expr] = sp.Symbol(expr)
                return variables[expr]

        elif isinstance(expr, (int, float)):
            return sp.Float(expr)
        else:
            raise ValueError(f"Unknown expression element: {expr}")
    
    # -------------------------------------------------------------
    sympy_expr = parse_expression(expr_tuple)
    simplified = sp.simplify(sympy_expr)
    if latex: 
        simplified = sp.latex(simplified, mode='equation')
    return simplified

# ------------------------------------------------------------------- EXPERIMENT RESULTS  -------------------------------------------------------------------------------------------
def aggregate_by_time_bins(logs, n_bins):
    """
    logs: list of dicts, each with keys
      ['time','train_rmse','val_rmse','nodes_count','diversity_var']
    n_bins: how many equally‐spaced time‐bins to form between the global min and max time
    Returns: DataFrame with columns
      time,
      median_*, mean_*, p25_*, p75_*  for each metric
    """
    # 1) build one big DataFrame of all runs
    dfs = []
    for run_idx, log in enumerate(logs):
        df = pd.DataFrame(log)
        df['run'] = run_idx
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # 2) compute global time range and bin edges
    tmin, tmax = all_df['time'].min(), all_df['time'].max()
    edges = np.linspace(tmin, tmax, n_bins + 1)

    # 3) assign each measurement to a bin index 0..n_bins-1
    #    (np.digitize returns 1..n_bins, so subtract 1)
    all_df['bin_idx'] = np.digitize(all_df['time'], edges) - 1
    # clamp any out‑of‑bounds into [0, n_bins-1]
    all_df['bin_idx'] = all_df['bin_idx'].clip(0, n_bins-1)

    # 4) group by that bin index and aggregate
    grouped = all_df.groupby('bin_idx')
    agg = grouped.agg(
        median_train_rmse    = ('train_rmse',    'median'),
        mean_train_rmse      = ('train_rmse',    'mean'),
        p25_train_rmse       = ('train_rmse',    lambda x: np.percentile(x, 25)),
        p75_train_rmse       = ('train_rmse',    lambda x: np.percentile(x, 75)),

        median_val_rmse      = ('val_rmse',      'median'),
        mean_val_rmse        = ('val_rmse',      'mean'),
        p25_val_rmse         = ('val_rmse',      lambda x: np.percentile(x, 25)),
        p75_val_rmse         = ('val_rmse',      lambda x: np.percentile(x, 75)),

        median_nodes_count   = ('nodes_count',   'median'),
        mean_nodes_count     = ('nodes_count',   'mean'),
        p25_nodes_count      = ('nodes_count',   lambda x: np.percentile(x, 25)),
        p75_nodes_count      = ('nodes_count',   lambda x: np.percentile(x, 75)),

        median_diversity_var = ('diversity_var', 'median'),
        mean_diversity_var   = ('diversity_var', 'mean'),
        p25_diversity_var    = ('diversity_var', lambda x: np.percentile(x, 25)),
        p75_diversity_var    = ('diversity_var', lambda x: np.percentile(x, 75)),
    )

    bin_centers_all = (edges[:-1] + edges[1:]) / 2.0
    agg = agg.reset_index()
    agg['time'] = agg['bin_idx'].map(lambda i: bin_centers_all[int(i)])
    agg = agg.drop(columns='bin_idx')
    cols = ['time'] + [c for c in agg.columns if c != 'time']
    return agg[cols]


def aggregate_by_generation(logs):
    """
    logs: list of dicts, each with keys
      ['generation','train_rmse','val_rmse','nodes_count','diversity_var']
    Returns: pd.DataFrame with one row per generation, columns
      median_*, mean_*, p25_*, p75_* for each metric.
    """
    # assume every log has the same generation vector
    gens = np.array(logs[0]['generation'])
    # stack each metric into an array shape (n_runs, n_gens)
    metrics = ['train_rmse','val_rmse','nodes_count','diversity_var']
    arrs = { m: np.vstack([log[m] for log in logs]) for m in metrics }

    data = {'generation': gens}
    for m, arr in arrs.items():
        data[f'median_{m}'] = np.median(arr, axis=0)
        data[f'mean_{m}']   = np.mean(arr,   axis=0)
        data[f'p25_{m}']    = np.percentile(arr, 25, axis=0)
        data[f'p75_{m}']    = np.percentile(arr, 75, axis=0)

    return pd.DataFrame(data)
    
def save_experiment_results(name, pfs, logs, n_bins=10, prefix='GP', suffix=None):
    """
    Record and save aggregated experiment data and Pareto front.

    Parameters
    ----------
    name : str
        Base name for saved files.
    pfs : list of list of tuples
        Pareto fronts for each run, each tuple is (rmse, complexity, time).
    logs : list of dicts
        Raw logs from each run (keys: 'time', 'generation', 'train_rmse', 'val_rmse', 'nodes_count', 'diversity_var').
    n_bins : int
        Number of time bins for aggregation.


    Returns
    -------
    df_time : pd.DataFrame
        DataFrame with time-based aggregated metrics.
    df_gen : pd.DataFrame
        DataFrame with generation-based aggregated metrics.

    """
    df_time = aggregate_by_time_bins(logs, n_bins)
    df_gen  = aggregate_by_generation(logs)
    pfs_complete = pf_rmse_comp_time(pfs)

    uname = platform.uname()
    vm = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True)
    metadata = {
        'system': uname.system,
        'node_name': uname.node,
        'release': uname.release,
        'version': uname.version,
        'machine': uname.machine,
        'processor': uname.processor,
        'cpu_count': cpu_count,
        'total_memory_bytes': vm.total,
        'python_version': platform.python_version(),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    df_meta = pd.DataFrame([metadata])

    os.makedirs(f'experiment_results/{name}', exist_ok=True)
    df_time.to_parquet(f'experiment_results/{name}/{prefix}_log_time{suffix}.pkl')
    df_gen.to_parquet(f'experiment_results/{name}/{prefix}_log_gen{suffix}.pkl')
    df_meta.to_parquet(f'experiment_results/{name}/{prefix}_metadata{suffix}.pkl')

    with open(f'experiment_results/{name}/{prefix}_pf{suffix}.pkl', 'wb') as f:
        pickle.dump(pfs_complete, f)

    return df_time, df_gen



def log_rmsetime(ag, name, filename="rmse_over_time.png", plot_train=True, plot_val=True):
    """
    Logs a plot of RMSE (median + IQR) over time to MLflow.
    
    Parameters:
        ag (pd.DataFrame): Output of aggregate_by_time_bins
        filename (str): Name for the saved plot
        plot_train (bool): Whether to plot training RMSE
        plot_val (bool): Whether to plot validation RMSE
    """
    plt.figure(figsize=(10, 6))

    if plot_train:
        plt.plot(ag['time'], ag['median_train_rmse'], label='Train Median', color='blue')
        plt.fill_between(ag['time'], ag['p25_train_rmse'], ag['p75_train_rmse'], alpha=0.2, label='Train IQR', color='blue')

    if plot_val:
        plt.plot(ag['time'], ag['median_val_rmse'], label='Val Median', color='orange')
        plt.fill_between(ag['time'], ag['p25_val_rmse'], ag['p75_val_rmse'], alpha=0.2, label='Val IQR', color='orange')

    plt.xlabel('Time (s)')
    plt.ylabel('RMSE')
    plt.title('RMSE over Time')
    plt.legend()
    plt.tight_layout()
    # Savefig to /images/<name> folder
    path = os.path.join('images', name, filename)
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

def log_rmsegen(df, name, filename="rmse_over_generations.png", plot_train=True, plot_val=True):
    """
    Logs a plot of RMSE (median + IQR) over generations to MLflow.
    
    Parameters:
        df (pd.DataFrame): A DataFrame with the required RMSE statistics by generation.
        filename (str): File name for the saved chart.
        plot_train (bool): Whether to plot training RMSE
        plot_val (bool): Whether to plot validation RMSE
    """
    plt.figure(figsize=(10, 6))

    if plot_train:
        plt.plot(df['generation'], df['median_train_rmse'], label='Train Median', color='blue')
        plt.fill_between(df['generation'], df['p25_train_rmse'], df['p75_train_rmse'], alpha=0.2, color='blue', label='Train IQR')

    if plot_val:
        plt.plot(df['generation'], df['median_val_rmse'], label='Val Median', color='orange')
        plt.fill_between(df['generation'], df['p25_val_rmse'], df['p75_val_rmse'], alpha=0.2, color='orange', label='Val IQR')

    plt.xlabel('Generation')
    plt.ylabel('RMSE')
    plt.title('RMSE over Generations')
    plt.legend()
    plt.tight_layout()
    path = os.path.join('images', name, filename)
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

def register_mlflow_charts(name, log_results, pfs, prefix='GP', suffix=None):
    df_time, df_gen = log_results
    # Check if images directory exists, if not create it
    os.makedirs('images', exist_ok=True)
    os.makedirs(f'images/{name}', exist_ok=True)

    log_rmsetime(df_time, name, filename=f'{prefix}_rmse_time_both_{suffix}.png', plot_train=True, plot_val=True)
    log_rmsetime(df_time, name, filename=f'{prefix}_rmse_time_train_{suffix}.png', plot_train=True, plot_val=False)
    log_rmsetime(df_time, name, filename=f'{prefix}_rmse_time_val_{suffix}.png', plot_train=False, plot_val=True)

    log_rmsegen(df_gen, name, filename=f'{prefix}_rmse_gen_both_{suffix}.png', plot_train=True, plot_val=True)
    log_rmsegen(df_gen, name, filename=f'{prefix}_rmse_gen_train_{suffix}.png', plot_train=True, plot_val=False)
    log_rmsegen(df_gen, name, filename=f'{prefix}_rmse_gen_val_{suffix}.png', plot_train=False, plot_val=True) 


def log_latex_as_image(latex_str, name, split_id, prefix='GP', suffix=None):
    """
    Renders a LaTeX string to an image and logs it to MLflow.
    
    Parameters:
        latex_str (str): The LaTeX representation of the expression
        filename (str): File name to log in MLflow
    """
    latex_clean = re.sub(r"\\begin\{.*?\}", "", latex_str)
    latex_clean = re.sub(r"\\end\{.*?\}", "", latex_clean).strip()
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"${latex_clean}$", fontsize=20, ha='center', va='center')
    ax.axis('off')
    fig.tight_layout()

    path = os.path.join('images', name, f'{prefix}_{split_id}_{suffix}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    mlflow.log_artifact(path)
    plt.close(fig)


# ------------------------------------------------------------------- OTHER GP2 -------------------------------------------------------------------------------------------
def save_experiment_results_v2(dataset_name: str,
                                 tuning_df: pd.DataFrame,
                                 best_hyperparams_map: dict,
                                 test_df: pd.DataFrame,
                                 pareto_fronts_data: list,
                                 logs_data: list,
                                 n_time_bins: int,
                                 prefix: str,
                                 suffix: str, 
                                 hp_base_dir: str = 'hp_results',
                                 test_base_dir: str = 'test_results',
                                 exp_base_dir: str = 'exp_results', 
                                 flag_exists: bool = False,
                                 ):
    """
    Saves all aggregated experiment artifacts for a given dataset.
    This function only writes data.
    """
    file_suffix_str = f"{suffix}" if suffix else ""

    current_hp_dir = Path(hp_base_dir) / dataset_name
    current_hp_dir.mkdir(parents=True, exist_ok=True)
    current_test_dir = Path(test_base_dir) / dataset_name
    current_test_dir.mkdir(parents=True, exist_ok=True)

    test_df_filename = current_test_dir / f'{prefix}_test_results{file_suffix_str}.parquet'

    if not flag_exists:
        tuning_df_filename = current_hp_dir / f'{prefix}_tuning_results{file_suffix_str}.parquet'
        hyperparams_filename = current_hp_dir / f'{prefix}_best_hyperparams{file_suffix_str}.pkl'

        filenames = [tuning_df_filename, hyperparams_filename, test_df_filename]
        for filename in filenames:
            if filename.exists():
                print(f"File {filename} already exists. Overwriting...")

        tuning_df.to_parquet(tuning_df_filename, index=False)

        with open(hyperparams_filename, 'wb') as f:
            pickle.dump(best_hyperparams_map, f)

    test_df.to_parquet(test_df_filename, index=False)

    log_results_summary = None
    if pareto_fronts_data or logs_data: 
         log_results_summary = save_experiment_results(
             dataset_name,
             pareto_fronts_data,
             logs_data,
             n_bins=n_time_bins,
             prefix=prefix,
             suffix=suffix,
         )
    
    return log_results_summary
