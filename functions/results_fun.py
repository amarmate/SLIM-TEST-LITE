import pickle
import os 
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tabulate import tabulate
import matplotlib.pyplot as plt
from slim_gsgp_lib.datasets.data_loader import *

datasets = [globals()[i] for i in globals() if 'load' in i][2:]
datasets = datasets[:12] + datasets[13:]  
dataset_dict = {}
df_datasets = {}
for i, dataset in enumerate(datasets):
    X,y = dataset()
    name = dataset.__name__.split('load_')[1]
    id = 'DA' + str(i).zfill(2)
    dataset_dict[name] = id
    df_datasets[name] = X.shape[0], X.shape[1]

# -------------------------------------------------------------------------------------------------------------------

def get_significance(p_value, ratio, detailed=False):
    if p_value >= 0.05:
        return 'NSD'
    elif ratio > 1:
        return '-' * (1 + int(p_value < 0.01) + int(p_value < 0.001)) if detailed else '-'
    else:
        return '+' * (1 + int(p_value < 0.01) + int(p_value < 0.001)) if detailed else '+'
    
def check_settings_availability(prefix_list):
    """
    Description:
    ------------
        Check if the settings for the given prefixes are available in the results folder.
    
    Parameters:
    ------------
        prefix_list: list of prefixes to compare (e.g., ['sc', 'scsm'])
    
    Returns:
    ----------
        possible: True if all settings are available, False otherwise
    """

    possible = True
    for prefix in prefix_list:
        settings, number = prefix.split('_')
        available = get_settings(number)
        if settings in available:
            continue
        else:
            print(f"Settings {prefix} not available. Choose from: {available}")
            possible = False
    
    return possible

def get_settings(n_prefix):
    """
    Description:
    ------------
        Get the settings available for a given prefix.

    Parameters:
    ------------
        n_prefix: the number of the prefix (e.g., '2') 

    Returns:
    ----------
        settings: list of settings available for the given prefix
    """

    settings = []
    for file in os.listdir(f'results/slim/DA00'):
        if not any(char.isdigit() for char in file):
            continue
        if file.split('_')[1].split('.pkl')[0] == str(n_prefix):
            settings.append(file.split('_')[0])
    return settings


def means_df(prefixes=['sc', 'scsm'],
             datasets=None,
             metrics=None, 
             best=False,
             decimals=3,
             table=False,
             detailed=False,
             sorted=False, 
             errors='ignore'):
    """
    Description:
    ------------
        A more general version of your means_df function. 
        Controlled via the 'metrics' dictionary, which maps
        a friendly name (e.g., 'rmse') to the key in the pickled results
        (e.g., 'rmse_compare' or 'mape', etc.).

        Example usage of 'metrics':
            metrics = {
                "rmse": "rmse_compare",
                "size": "size",
                "time": "time",
                "mape": "mape"
            }

        If you only want RMSE and size:
            metrics = {
                "rmse": "rmse_compare",
                "size": "size",
            }

    Parameters:
    ------------
        prefixes: list of prefixes to compare (e.g., ['sc', 'scsm'])
        datasets: list of datasets to include (e.g., ['DA00', 'DA01'])
        metrics: dictionary of metrics to include (e.g., {"rmse": "rmse_compare"})
        best: if True, will also compute the best algorithm for each dataset
        decimals: number of decimals to round to
        table: if True, print the result as a table
        detailed: if True, will show detailed significance (e.g., '++' or '--')
        sorted: if True, will sort the table by the RMSE ratio
        errors: 'ignore' to ignore errors, 'raise' to raise errors
    
    Returns:
    ----------
        DataFrame with the results or None if table=True

    """
    if check_settings_availability(prefixes) is False:
        return None

    if metrics is None:
        # Default to just RMSE
        metrics = {"rmse": "rmse_compare"}  

    global dataset_dict
    if datasets is None:
        datasets = dataset_dict.keys()

    sig_test_data = {}
    data_rows = {m: [] for m in metrics.keys()}
    data_rows_std = {m: [] for m in metrics.keys()}

    # 1) Collect data from pickles
    for dataset in datasets:
        if dataset not in sig_test_data:
            sig_test_data[dataset] = {}

        for prefix in prefixes:
            if prefix not in sig_test_data[dataset]:
                sig_test_data[dataset][prefix] = {}

            with open(f"results/slim/{dataset_dict[dataset]}/{prefix}.pkl", "rb") as f:
                results = pickle.load(f)

            algorithms = results[list(metrics.values())[0]].keys()
            for algo in algorithms:
                if algo not in sig_test_data[dataset][prefix]:
                    sig_test_data[dataset][prefix][algo] = {}

                for m_name, m_key in metrics.items():
                    metric_values = results[m_key][algo] if m_key in results else None
                    if metric_values is None:
                        continue 

                    data_rows[m_name].append({
                        "Dataset": dataset,
                        "Algorithm": algo,
                        f"{m_name}_{prefix}": np.mean(metric_values)
                    })
                    data_rows_std[m_name].append({
                        "Dataset": dataset,
                        "Algorithm": algo,
                        f"{m_name}_std_{prefix}": np.std(metric_values)
                    })

                    sig_test_data[dataset][prefix][algo][m_name] = np.array(metric_values)

    # 2) Build pivot tables for each metric
    df_dict = {}
    for m_name in metrics.keys():
        df_means = pd.DataFrame(data_rows[m_name])
        if not df_means.empty:
            df_means = df_means.pivot_table(
                index=["Dataset", "Algorithm"],
                values=[f"{m_name}_{p}" for p in prefixes],
                aggfunc="first"
            ).reset_index()
            df_means = df_means[
                ["Dataset", "Algorithm"] + [f"{m_name}_{p}" for p in prefixes]
            ]
        else:
            df_means = pd.DataFrame(columns=["Dataset", "Algorithm"] + [f"{m_name}_{p}" for p in prefixes])

        df_stds = pd.DataFrame(data_rows_std[m_name])
        if not df_stds.empty:
            df_stds = df_stds.pivot_table(
                index=["Dataset", "Algorithm"],
                values=[f"{m_name}_std_{p}" for p in prefixes],
                aggfunc="first"
            ).reset_index()
            df_stds = df_stds[
                ["Dataset", "Algorithm"] + [f"{m_name}_std_{p}" for p in prefixes]
            ]
        else:
            df_stds = pd.DataFrame(columns=["Dataset", "Algorithm"] + [f"{m_name}_std_{p}" for p in prefixes])

        df_merged = pd.merge(df_means, df_stds, on=["Dataset", "Algorithm"], how="outer")
        df_dict[m_name] = df_merged

    # 3) Compute ratio columns if exactly 2 prefixes are given (or you can adapt for more)
    if len(prefixes) == 2:
        for m_name in metrics.keys():
            col_a = f"{m_name}_{prefixes[0]}"
            col_b = f"{m_name}_{prefixes[1]}"
            if col_a in df_dict[m_name].columns and col_b in df_dict[m_name].columns:
                df_dict[m_name][f"ratio_{m_name}"] = (
                    df_dict[m_name][col_a] / df_dict[m_name][col_b]
                )

    # 4) Compute significance using Wilcoxon for each dataset, algo, metric,
    for m_name in metrics.keys():
        df_dict[m_name].set_index(["Dataset", "Algorithm"], inplace=True)

    for dataset in sig_test_data.keys():
        for algo in sig_test_data[dataset][prefixes[0]].keys():
            for m_name in metrics.keys():
                base_data = sig_test_data[dataset][prefixes[0]][algo].get(m_name, None)
                if base_data is None:
                    continue
                for px in prefixes[1:]:
                    comp_data = sig_test_data[dataset][px][algo].get(m_name, None)
                    if comp_data is None:
                        continue
                    min_len = min(len(base_data), len(comp_data))
                    base_vals = base_data[:min_len]
                    comp_vals = comp_data[:min_len]
                    try:
                        _, p_value = wilcoxon(
                            np.round(comp_vals - base_vals, decimals=8),
                            zero_method="pratt",
                            alternative="two-sided",
                            method="approx"
                        )
                    except Exception as e:
                        print(f"Error in significance test: {e}, {dataset}, {algo}, prefix {px}") if errors == 'raise' else None
                        p_value = 1
                    ratio_col = f"ratio_{m_name}"
                    ratio_val = None
                    if ratio_col in df_dict[m_name].columns:
                        ratio_val = df_dict[m_name].loc[(dataset, algo), ratio_col]
                    significance = get_significance(
                        p_value, ratio_val if ratio_val is not None else 1.0,
                        detailed=detailed
                    )
                    df_dict[m_name].loc[(dataset, algo), f"{m_name}_significance"] = significance
    for m_name in metrics.keys():
        df_dict[m_name].reset_index(inplace=True)

    # 5) Combine all metric frames into one “combined” DataFrame
    df_combined = None
    for i, (m_name, df_m) in enumerate(df_dict.items()):
        if i == 0:
            df_combined = df_m
        else:
            df_combined = pd.merge(df_combined, df_m, on=["Dataset", "Algorithm"], how="outer")

    if df_combined is not None:
        df_combined = df_combined.round(decimals)

    # 6) “Best” logic: if best=True, call a separate function that picks the best for each dataset
    if best and df_combined is not None:
        return _means_df_best(df_dict, prefixes, metrics, decimals, table, detailed)

    # 7) If we want to print as a table
    if table and df_combined is not None:
        for m_name in metrics.keys():
            for px in prefixes:
                mean_col = f"{m_name}_{px}"
                std_col = f"{m_name}_std_{px}"
                if mean_col in df_combined.columns and std_col in df_combined.columns:
                    df_combined[f"{m_name}_{px}"] = (
                        df_combined[mean_col].astype(str)
                        + " ± "
                        + df_combined[std_col].astype(str)
                    )
        df_combined = df_combined[
            [c for c in df_combined.columns if not c.endswith("_std_" + prefixes[0]) and not c.endswith("_std_" + prefixes[1])]
        ]
        df_combined.sort_values("ratio_rmse", inplace=True) if sorted else None
        print(tabulate(df_combined, headers="keys", tablefmt="fancy_grid"))
        return None

    return df_combined.set_index(["Dataset", "Algorithm"])


# -------------------------------------------------------------------------------------------------------------------

def _means_df_best(
    df_dict,       # dict of DataFrames for each metric, e.g. df_dict["rmse"], df_dict["size"], ...
    prefixes,      # e.g. ["sc", "scsm"]
    metrics,       # e.g. {"rmse": "rmse_compare", "size": "size", "time": "time"}
    decimals=3,
    table=False,
    detailed=False
):
    """
    Description:
    ------------
    This function is called by means_df() when best=True.
    It is responsible for finding the best algorithm for each dataset.
    The logic is as follows:
        1) For each dataset, find the best algorithm in terms of RMSE for each prefix independently.
        2) Gather means (and std) for that chosen algorithm in each prefix, for *all* metrics.
        3) Compare the repeated measures for the chosen prefix0-algo vs prefix1-algo with Wilcoxon,
        for each metric. 
        4) Produce a single row per dataset in the final DataFrame (with dataset as the index).
        5) Optionally, print the result as a table.

    Parameters:
    ------------
        df_dict: dict of DataFrames for each metric, e.g. df_dict["rmse"], df_dict["size"], ...
        prefixes: e.g. ["sc", "scsm"]
        metrics: e.g. {"rmse": "rmse_compare", "size": "size", "time": "time"}
        decimals: number of decimals to round to
        table: if True, print the result as a table
        detailed: if True, will show detailed significance (e.g., '++' or '--')

    Returns:
    ----------
        DataFrame with the results or None if table=True
    """

    if "rmse" not in metrics:
        raise ValueError("To pick the best algorithm, 'rmse' must be one of the keys in 'metrics'.")

    df_rmse = df_dict["rmse"].copy()
    if "Dataset" not in df_rmse.columns:
        df_rmse.reset_index(inplace=True)
    best_rows = []

    def load_repeated_measures(dataset, prefix, algo, metric_key):
        """
        Loads repeated measure data from pickled results for the given dataset, prefix, and algo.
        metric_key might be 'rmse_compare' or 'size' or 'time', etc.
        Returns np.array of repeated-measure values.
        """
        dataset_id = dataset_dict[dataset]
        with open(f"results/slim/{dataset_id}/{prefix}.pkl", "rb") as f:
            results = pickle.load(f)
        return np.array(results[metric_key][algo])

    # 1) For each dataset, find the “best” algorithm for each prefix (lowest RMSE).
    all_datasets = df_rmse["Dataset"].unique()
    if len(prefixes) != 2:
        print("Note: This code is specialized for exactly 2 prefixes (e.g., sc vs scsm).")
        print("If you have more prefixes, further modifications are needed.")

    # For easier referencing, name them p0, p1
    p0, p1 = prefixes[0], prefixes[1]

    for ds in all_datasets:
        df_sub = df_rmse[df_rmse["Dataset"] == ds]

        col_p0 = f"rmse_{p0}"
        col_p1 = f"rmse_{p1}"

        if col_p0 not in df_sub.columns or col_p1 not in df_sub.columns:
            continue

        idxmin_p0 = df_sub[col_p0].idxmin()
        row_p0 = df_sub.loc[idxmin_p0]
        best_algo_p0 = row_p0["Algorithm"]

        idxmin_p1 = df_sub[col_p1].idxmin()
        row_p1 = df_sub.loc[idxmin_p1]
        best_algo_p1 = row_p1["Algorithm"]
        row_data = {
            "Dataset": ds,
            f"best_algo_{p0}": best_algo_p0,
            f"best_algo_{p1}": best_algo_p1
        }
        for m_name, m_key in metrics.items():
            df_m = df_dict[m_name].copy()
            if "Dataset" not in df_m.columns:
                df_m.reset_index(inplace=True)
            row_p0_m = df_m[
                (df_m["Dataset"] == ds) & (df_m["Algorithm"] == best_algo_p0)
            ]
            row_p1_m = df_m[
                (df_m["Dataset"] == ds) & (df_m["Algorithm"] == best_algo_p1)
            ]
            mean_col_p0 = f"{m_name}_{p0}"
            mean_col_p1 = f"{m_name}_{p1}"
            std_col_p0 = f"{m_name}_std_{p0}"
            std_col_p1 = f"{m_name}_std_{p1}"

            val_p0 = row_p0_m[mean_col_p0].iloc[0] if mean_col_p0 in row_p0_m.columns and len(row_p0_m) > 0 else np.nan
            val_p1 = row_p1_m[mean_col_p1].iloc[0] if mean_col_p1 in row_p1_m.columns and len(row_p1_m) > 0 else np.nan
            row_data[f"{m_name}_{p0}"] = val_p0
            row_data[f"{m_name}_{p1}"] = val_p1
            std_p0_val = row_p0_m[std_col_p0].iloc[0] if std_col_p0 in row_p0_m.columns and len(row_p0_m) > 0 else np.nan
            std_p1_val = row_p1_m[std_col_p1].iloc[0] if std_col_p1 in row_p1_m.columns and len(row_p1_m) > 0 else np.nan
            row_data[f"{m_name}_std_{p0}"] = std_p0_val
            row_data[f"{m_name}_std_{p1}"] = std_p1_val

            if not np.isnan(val_p0) and not np.isnan(val_p1) and val_p1 != 0:
                row_data[f"ratio_{m_name}"] = val_p0 / val_p1
            else:
                row_data[f"ratio_{m_name}"] = np.nan

            try:
                arr_p0 = load_repeated_measures(ds, p0, best_algo_p0, m_key)
                arr_p1 = load_repeated_measures(ds, p1, best_algo_p1, m_key)

                min_len = min(len(arr_p0), len(arr_p1))
                arr_p0 = arr_p0[:min_len]
                arr_p1 = arr_p1[:min_len]

                from scipy.stats import wilcoxon
                _, p_value = wilcoxon(
                    np.round(arr_p0 - arr_p1, decimals=8),
                    zero_method="pratt",
                    alternative="two-sided",
                    method="approx"
                )
                significance = get_significance(
                    p_value, row_data[f"ratio_{m_name}"], detailed=detailed
                )
                row_data[f"{m_name}_significance"] = significance
            except Exception as e:
                row_data[f"{m_name}_significance"] = "!"

        best_rows.append(row_data)
    best_df = pd.DataFrame(best_rows)
    if best_df.empty:
        print("No best data found (best_df is empty).")
        return best_df
    numeric_cols = best_df.select_dtypes(include=[np.number]).columns
    best_df[numeric_cols] = best_df[numeric_cols].round(decimals)
    best_df.set_index("Dataset", inplace=True)
    if table:
        for m_name in metrics.keys():
            for px in prefixes:
                mean_col = f"{m_name}_{px}"
                std_col = f"{m_name}_std_{px}"
                if mean_col in best_df.columns and std_col in best_df.columns:
                    best_df[mean_col] = (
                        best_df[mean_col].astype(str)
                        + " ± "
                        + best_df[std_col].astype(str)
                    )

        std_cols = [c for c in best_df.columns if "_std_" in c]
        best_df.drop(columns=std_cols, inplace=True, errors="ignore")
        best_df.sort_values("ratio_rmse", inplace=True)
        print(tabulate(best_df, headers="keys", tablefmt="fancy_grid", floatfmt=f".{decimals}f"))
        return None

    return best_df.sort_values("ratio_rmse")

# -------------------------------------------------------------------------------------------------------------------

def summary_results(prefixes, metrics=None, datasets=None):
    """
    Description:    
    ------------
    This function is responsible for summarizing the results of the SLIM experiments.
    It will print a table with the counts of significance levels (+, NSD, -) for each metric.
    It will also print the median and mean ratios for each metric.
    
    Parameters:
    ------------
    prefixes: list of prefixes to compare (e.g., ['sc', 'scsm'])
    datasets: list of datasets to include (e.g., ['DA00', 'DA01'])
    metrics: dictionary of metrics to include (e.g., {"rmse": "rmse_compare"})

    Returns:
    ----------
    None
    """

    if metrics is None:
        metrics = {"rmse": "rmse_compare", "size": "size", "time": "time"}
    df_normal = means_df(
        prefixes=prefixes,
        datasets=datasets,
        metrics=metrics,
        best=False,
        decimals=3,
        table=False,
        detailed=False,
        sorted=False,
    )

    df_best = means_df(
        prefixes=prefixes,
        datasets=datasets,
        metrics=metrics,
        best=True,
        decimals=3,
        table=False,
        detailed=False,
    )
    data = {}
    for m in metrics.keys():
        data[m] = {
            'sig_count': df_normal[f'{m}_significance'].value_counts(),
            'ratio': df_normal[f'ratio_{m}'].dropna(),
            'sig_count_best': df_best[f'{m}_significance'].value_counts(),
            'ratio_best': df_best[f'ratio_{m}'].dropna(),
        }
    df_significance = pd.DataFrame({
        metric: values['sig_count'] for metric, values in data.items()
    }).T.fillna(0)

    df_significance_best = pd.DataFrame({
        metric: values['sig_count_best'] for metric, values in data.items()
    }).T.fillna(0)

    df_significance['Median Ratio'] = [values['ratio'].median() for values in data.values()]
    df_significance_best['Median Ratio'] = [values['ratio_best'].median() for values in data.values()]
    df_significance['Mean Ratio'] = [values['ratio'].mean() for values in data.values()]
    df_significance_best['Mean Ratio'] = [values['ratio_best'].mean() for values in data.values()]

    for df in [df_significance, df_significance_best]:
        numeric_df = df[['+', 'NSD', '-']].astype(float) 
        row_sums = numeric_df.sum(axis=1) 

        for col in ['+', 'NSD', '-']:
            if col in df.columns:
                df[col] = numeric_df[col].astype(int).astype(str) + " (" + (
                    100 * numeric_df[col] / row_sums
                ).astype(int).astype(str) + "%)"

    table_normal_str = tabulate(
        df_significance.reset_index(),
        headers=["Metric"] + list(df_significance.columns),
        tablefmt="grid",
        showindex=False,
    )
    table_best_str = tabulate(
        df_significance_best.reset_index(),
        headers=["Metric"] + list(df_significance_best.columns),
        tablefmt="grid",
        showindex=False,
    )
    
    # Calculate table width for alignment
    table_width = max(
        max(len(line) for line in table_normal_str.splitlines()),
        max(len(line) for line in table_best_str.splitlines()),
    )

    # Center the titles relative to the table width
    title_normal = "All Significance Counts and Ratios".center(table_width)
    title_best = "Best Significance Counts and Ratios".center(table_width)

    # Add titles and split tables into lines
    lines_normal = [title_normal] + table_normal_str.splitlines()
    lines_best = [title_best] + table_best_str.splitlines()

    # Ensure both tables have the same number of lines by padding the shorter one
    max_len = max(len(lines_normal), len(lines_best))
    lines_normal += [" " * table_width] * (max_len - len(lines_normal))
    lines_best += [" " * table_width] * (max_len - len(lines_best))

    # Combine lines side by side with spacing
    side_by_side_output = [
        f"{ln:<{table_width}}    {lb}" for ln, lb in zip(lines_normal, lines_best)
    ]
    print("\n".join(side_by_side_output))



    fig, axs = plt.subplots(1, len(data), figsize=(6 * len(data), 6), sharey=False)
    if len(data) == 1:
        axs = [axs]

    for metric, values in data.items():
        q1, q3 = np.percentile(values['ratio'], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[metric]['ratio'] = values['ratio'][
            (values['ratio'] > lower_bound) & (values['ratio'] < upper_bound)
        ]

        q1, q3 = np.percentile(values['ratio_best'], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[metric]['ratio_best'] = values['ratio_best'][
            (values['ratio_best'] > lower_bound) & (values['ratio_best'] < upper_bound)
        ]

    for ax, (metric, values) in zip(axs, data.items()):
        ax.boxplot([values['ratio'], values['ratio_best']], tick_labels=["Normal", "Best"])
        ax.set_title(f"{metric} ratios")
        ax.set_ylabel("Ratio")
        ax.grid(axis="y")

    # Adjust layout to ensure proper spacing
    plt.tight_layout()
    plt.show()