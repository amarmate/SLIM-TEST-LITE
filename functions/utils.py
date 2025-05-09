

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
    The function will select:
    - Top 4 individuals with the best RMSE,
    - Top 2 individuals with the lowest complexity,
    - First individual with the least time taken.
    
    Parameters
    ----------
    points : list of tuples (rmse, comp, time, id)
        A list of individuals from the Pareto front. Each individual is represented as 
        (RMSE, complexity, time, trial_id).

    Returns
    -------
    list
        A Pareto front containing the selected individuals based on the criteria.
    """

    pareto = []
    for i, (rmse1, comp1, time1, id1) in enumerate(points):
        dominated = False
        for j, (rmse2, comp2, time2, id2) in enumerate(points):
            if j != i and (rmse2 <= rmse1 and comp2 <= comp1 and time2 <= time1) and (rmse2 < rmse1 or comp2 < comp1 or time2 < time1):
                dominated = True
                break
        if not dominated:
            pareto.append((rmse1, comp1, time1, id1))

    pareto.sort(key=lambda x: (x[0], x[1], x[2]))
    return pareto


def save_tuning_results(name: str,
                        split_id: int,
                        df_tr: pd.DataFrame,
                        best_hyperparams: dict,
                        base_dir: str = 'hp_results'):
    """
    Persist hyperparameter tuning results and best‐params per split.

    - Stores/updates per‐split trial log in <base_dir>/<name>/tr_results.pkl
    - Stores/updates best hyperparameters dict in <base_dir>/<name>/params.pkl

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

    # --- update trial‐results log ---
    tr_path = os.path.join(dir_path, 'tr_results.pkl')
    if os.path.exists(tr_path):
        # append new trials to existing log
        previous_tr = pd.read_pickle(tr_path)
        combined = pd.concat([previous_tr, df_tr], ignore_index=True)
        combined.to_pickle(tr_path)
    else:
        # first time: write full df
        df_tr.to_pickle(tr_path)

    # --- update best‐hyperparams per split ---
    params_path = os.path.join(dir_path, 'params.pkl')
    if os.path.exists(params_path):
        previous_params = pickle.load(open(params_path, 'rb'))
        previous_params[split_id] = best_hyperparams
        pickle.dump(previous_params, open(params_path, 'wb'))
    else:
        # first time: create dict with this split
        new_params = {split_id: best_hyperparams}
        pickle.dump(new_params, open(params_path, 'wb'))
